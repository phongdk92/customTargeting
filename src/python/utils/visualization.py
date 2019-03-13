#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Thu Sep 20 10:57:42 2018

@author: phongdk
"""
import os
import logging
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report, precision_recall_curve, \
    average_precision_score, recall_score, precision_score, f1_score
from sklearn.manifold import TSNE
import tensorflow as tf
from tensorflow.keras.utils import to_categorical

# import matplotlib
# matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns
import warnings
warnings.filterwarnings("ignore")

LOGGER = logging.getLogger(__name__)

def out_statistics(Y_test, Y_pred, path_figure='.', filename="temp", sample_weight=None, sys_show=False):
    LOGGER.info(accuracy_score(Y_test, Y_pred))
    target_names = [str(i) for i in range(len(np.unique(Y_test)))]
    '''
    don't know why uncomment those snippet of below code make model predict different labels even using the same model,
    perhaps matplotlib make something changes in LGBM library :(((
    '''
    plt.close()
    cm = confusion_matrix(Y_test, Y_pred)
    df_cm = pd.DataFrame(cm, index=target_names, columns=target_names)
    '''plot confusion matrix'''
    heatmap = sns.heatmap(df_cm, annot=True, fmt="d")
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    heatmap.yaxis.set_ticklabels(heatmap.yaxis.get_ticklabels(), rotation=0, ha='right')
    heatmap.xaxis.set_ticklabels(heatmap.xaxis.get_ticklabels(), rotation=0, ha='right')
    plt.title("Confusion matrix")
    plt.savefig(os.path.join(path_figure, filename))
    if sys_show:
        plt.show()
    LOGGER.info(classification_report(Y_test, Y_pred, target_names=target_names, sample_weight=sample_weight))


def visualize_data(x_test, y_test, filename="temp"):
    import colorsys
    from sklearn.decomposition.pca import PCA
    pca_50 = PCA(n_components=50)
    x_test = pca_50.fit_transform(x_test)
    # data2 = pd.concat([x_test,y_test], axis =1)
    tsne = TSNE(n_components=2, verbose=0, perplexity=30, n_iter=1000)
    tsne_results = tsne.fit_transform(x_test, y_test)

    N = len(np.unique(y_test))
    LOGGER.info('unique {}'.format(N))
    HSV_tuples = [(x * 1.0 / N, 0.5, 0.5) for x in range(N)]
    RGB_tuples = map(lambda x: colorsys.hsv_to_rgb(*x), HSV_tuples)
    plt.scatter(tsne_results[:, 0], tsne_results[:, 1], c=RGB_tuples)
    plt.savefig(os.path.join(path_figure, filename))
    plt.show()


def out_Precision_Recall_Curve(Y_test, Y_score, filename="temp"):
    #    out_Precision_Recall_Curve(Y_gender_test[:,0], Y_gender_prob[:,0])
    precision, recall, _ = precision_recall_curve(Y_test, Y_score)
    average_precision = average_precision_score(Y_test, Y_score)
    LOGGER.info('Average precision-recall score: {0:0.2f}'.format(average_precision))
    plt.step(recall, precision, color='b', alpha=0.2, where='post')
    plt.fill_between(recall, precision, step='post', alpha=0.2, color='b')

    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.ylim([0.0, 1.05])
    plt.xlim([0.0, 1.0])
    plt.show()


def out_Extension_Precision_Recall_Curve(Y_test, Y_score, sample_weight=None, filename="temp"):
    n_classes = len(np.unique(Y_test))
    Y_test = to_categorical(Y_test)
    precision = dict()
    recall = dict()
    average_precision = dict()
    lines = []
    labels = []
    for i in range(n_classes):
        precision[i], recall[i], threshold = precision_recall_curve(Y_test[:, i], Y_score[:, i],
                                                                    sample_weight=sample_weight)
        average_precision[i] = average_precision_score(Y_test[:, i], Y_score[:, i], sample_weight=sample_weight)
    LOGGER.info('threshold {}'.format(threshold))

    for i in range(n_classes):
        l, = plt.plot(recall[i], precision[i], lw=2)
        lines.append(l)
        labels.append('Precision Recall for class {0} (area = {1:0.2f})'
                      ''.format(i, average_precision[i]))

    fig = plt.gcf()
    fig.subplots_adjust(bottom=0.25)
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title('Extension of Precision-Recall curve to multi-class')
    plt.legend(lines, labels, loc='upper right', prop=dict(size=14))
    #    plt.legend(lines, labels, loc=(0, -.38), prop=dict(size=14))
    #    plt.savefig(os.path.join(path_figure,filename))
    plt.show()


def visualize_distribution(df, title="", filename=''):
    ax = sns.countplot(data=df, x="gender", hue="age_group")
    #   plt.title("Gender distribution")
    plt.yticks(range(0, 40000, 5000))
    plt.xticks(range(2), ("Male", "Female"))
    plt.title(title)
    #    plt.xticks(range(6), ("0-18","19-24","25-34", "35-44", "45-54", "55+"))
    plt.legend(["0-17", "18-24", "25-34", "35-44", "45-54", "55+"])
    plt.savefig(os.path.join(path_figure, filename))
    plt.show()


def plot_mean_and_CI(threshold, mean, lb, ub, color_mean=None, color_shading=None):
    # plot the shaded range of the confidence intervals
    plt.fill_between(threshold, ub, lb, alpha=.5)  # , color=color_shading)
    # plot the mean on top
    plt.plot(threshold, mean)  # , color_mean)


def plot_business_values(threshold, precision, recall, volume, business_values, specific_classes, MIN_THRESHOLD_CLASS,
                         path_figure='.', filename='temp', sys_show=False, show_score=False, ylim_maxBV=1.5):
    plt.close()
    total_BV = np.sum(business_values, axis=1)
    max_BV_pos = np.argmax(total_BV)
    max_BV = total_BV[max_BV_pos]
    optimal_threshold = threshold[max_BV_pos]

    labels = {'Precision': [], 'Recall': [], '1-CDF': [], 'Precision-Recall': [], 'BV': []}
    for c in specific_classes:  # (nclass):
        for key in labels.keys():
            if key != 'BV':
                labels[key].append('{} for class {}'.format(key, c))
    #                else:
    #                    labels['BV'].append('Class {} -- T: {}, BV: {}'.format(c,
    #                          threshold[np.argmax(business_values[:,c])], round(np.max(business_values[:,c]),2)))

    nrow, ncol = 3, 2
    gs = gridspec.GridSpec(nrow, ncol)

    '''plot business value '''
    plt.subplot(gs[2, :], xlabel='Threshold', ylabel='Additional_Money_O',
                title='Business Value for each class and Total Business Value')
    for c in specific_classes:  # range(nclass):
        plt.plot(threshold, business_values[:, c], lw=2)
    plt.plot(threshold, total_BV, lw=2, color='r')
    plot_marker_point(optimal_threshold, max_BV)

    pos_maxBV_each_class = {}  # position with max business value for each class
    LOGGER.info('Min threshold for each class : '.format(MIN_THRESHOLD_CLASS))
    for c in specific_classes:  # range(nclass):
        segment_threshold = max(MIN_THRESHOLD_CLASS, threshold[np.argmax(business_values[:, c])])
        pos = np.argmin(abs(segment_threshold - threshold))
        # pos = np.argmax(business_values[:,c])
        plot_marker_point(threshold[pos], round(business_values[pos, c], 2))  # plot best threshold for each class
        pos_maxBV_each_class[c] = pos
        labels['BV'].append('Class {} -- T: {}, BV: {}'.format(c,
                                                               threshold[pos],
                                                               round(np.max(business_values[pos, c]), 2)))
    labels['BV'].append('Total -- T: {}, BV: {}'.format(optimal_threshold, round(max_BV, 2)))
    plt.xlim([0.0, 1.0])
    plt.ylim([np.min(total_BV), ylim_maxBV])
    plt.legend(labels['BV'], loc='upper right', prop=dict(size=12))

    '''plot TPRV'''
    xlabels = ['recall'] + ['threshold'] * 3
    ylabels = ['precision'] * 2 + ['volume'] + ['recall']
    xdata = [recall, threshold, threshold, threshold]
    ydata = [precision, precision, volume, recall]
    legend = {}  # dict to map each subplot to its legend
    for xlabel, ylabel, key in zip(xlabels, ylabels, ['Precision-Recall', 'Precision', '1-CDF',
                                                      'Recall']):  # cannot use labels.keys() since it releases data in different order
        legend[xlabel + ylabel] = labels[key]
    for i in range(nrow - 1):
        for j in range(ncol):
            xlabel = xlabels[i * 2 + j]
            ylabel = ylabels[i * 2 + j]
            try:
                plt.subplot(gs[i, j], sharex=ax1, sharey=ax1, xlabel=xlabel, ylabel=ylabel)
            except:
                ax1 = plt.subplot(gs[i, j], xlabel=xlabel, ylabel=ylabel)
            lx, ly = [], []
            for c in specific_classes:  # range(nclass):
                pos = pos_maxBV_each_class[c]
                try:
                    l = plt.plot(xdata[i * 2 + j][:, c], ydata[i * 2 + j][:, c],
                                 lw=2)  # xdata=recall[:,c]  for each class c
                    x, y = xdata[i * 2 + j][pos, c], ydata[i * 2 + j][pos, c]
                except:
                    l = plt.plot(xdata[i * 2 + j], ydata[i * 2 + j][:, c], lw=2)  # xdata=threshold
                    x, y = xdata[i * 2 + j][pos], ydata[i * 2 + j][pos, c]
                # LOGGER.info(l.get_color())
                lx.append(x)
                ly.append(y)

            #                    if (xlabel == 'threshold'):
            #                        if (ylabel == 'precision'):
            #                            #interval = precision_interval[:,c,1] - precision[:,c]
            #                            plot_mean_and_CI(precision[:,c], precision_interval[:,c,0], precision_interval[:,c,1],
            #                                             color_mean=l[0].get_color(), color_shading=l[0].get_color())
            #                        elif (ylabel == 'recall'):
            #                            plot_mean_and_CI(precision[:,c], recall_interval[:,c,0], recall_interval[:,c,1],
            #                                             color_mean=l[0].get_color(), color_shading=l[0].get_color())
            # LOGGER.info('color', l[0].get_color())
            for (x, y) in zip(lx, ly):
                plot_marker_point(x, y, show_score=show_score)
            plt.xlim([0.0, 1.0])
            plt.ylim([0.0, 1.05])
            plt.legend(legend[xlabel + ylabel], loc='upper right', prop=dict(size=12))

    LOGGER.info('Save figure to : {}'.format(os.path.join(path_figure, filename)))
    #        ##plt.savefig(os.path.join(path_figure, filename), bbox_inches='tight')
    fig = plt.gcf()
    fig.set_size_inches((18, 12), forward=False)
    fig.savefig(os.path.join(path_figure, filename))

    if sys_show:
        manager = plt.get_current_fig_manager()  # https://stackoverflow.com/questions/32428193/saving-matplotlib-graphs-to-image-as-full-screen/32428266
        manager.window.showMaximized()  ## QT backend
        # manager.resize(*manager.window.maxsize())  #
        plt.show()
    return optimal_threshold, max_BV, segment_threshold


def plot_marker_point(x, y, color='#2B385A', show_score=False):
    plt.plot([x], [y], marker='X', color=color)
    plt.plot([x, x], [0, y], '--', color=color)
    plt.plot([0, x], [y, y], '--', color=color)
    if show_score:
        plt.text(x - 0.02, 0.07, str(round(x, 2)), fontsize=10, rotation=90)
        plt.text(0.02, y + 0.02, str(round(y, 2)), fontsize=10)
    # plt.xticks(rotation=45)


# def show_TPRV(Y_test, Y_prob, nclass, distribution, sample_weight = None, known_type = 'threshold',
#              known_value=0.4, known_class=0,  filename='temp'):
#     '''
#     Given a value of one metric (precsion, recall or volumne) or threshold, find other metrics
#     '''
#     threshold, precision, recall, volume, _, _ = compute_precision_recall_volume_BV(Y_test, Y_prob, nclass,
#                                                                                                distribution, sample_weight)

def plot_TPRV(threshold, precision, recall, volume, nclass, known_type='threshold', known_value=0.4, known_class=0,
              filename='temp'):
    '''get all other metrics'''
    # metrics = [threshold, avg_precision, avg_recall, avg_volume]
    metrics = [threshold, precision[:, known_class], recall[:, known_class], volume[:, known_class]]
    type_metric = {'threshold': 0, 'precision': 1, 'recall': 2, 'volume': 3}
    position = np.argmin(abs(metrics[type_metric[known_type]] - known_value))
    LOGGER.info('--------------- Known type: {},  with value is: {} \nOther metrics are:'.format(known_type, known_value))

    for key, value in type_metric.items():
        if key != known_type:
            LOGGER.info("{} , {} ".format(key, metrics[value][position]))

    labels = {'Precision': [], 'Recall': [], '1-CDF': [], 'Precision-Recall': []}
    for c in range(nclass):
        for key in labels.keys():
            labels[key].append('{} for class {}'.format(key, c))

    nrow, ncol = 2, 2
    gs = gridspec.GridSpec(nrow, ncol)
    xlabels = ['recall'] + ['threshold'] * 3
    ylabels = ['precision'] * 2 + ['volume'] + ['recall']
    xdata = [recall, threshold, threshold, threshold]
    ydata = [precision, precision, volume, recall]

    legend = {}  # dict to map each subplot to its legend
    for xlabel, ylabel, key in zip(xlabels, ylabels, ['Precision-Recall', 'Precision', '1-CDF',
                                                      'Recall']):  # cannot use labels.keys() since it releases data in different order
        legend[xlabel + ylabel] = labels[key]
    for i in range(nrow):
        for j in range(ncol):
            xlabel = xlabels[i * 2 + j]
            ylabel = ylabels[i * 2 + j]
            try:
                plt.subplot(gs[i, j], sharex=ax1, sharey=ax1, xlabel=xlabel, ylabel=ylabel)
            except:
                ax1 = plt.subplot(gs[i, j], xlabel=xlabel, ylabel=ylabel)
            for c in range(nclass):
                try:
                    plt.plot(xdata[i * 2 + j][:, c], ydata[i * 2 + j][:, c],
                             lw=2)  # xdata=recall[:,c]  for each class c
                except:
                    plt.plot(xdata[i * 2 + j], ydata[i * 2 + j][:, c], lw=2)  # xdata=threshold
            x, y = metrics[type_metric[xlabel]][position], metrics[type_metric[ylabel]][position]
            plot_marker_point(x, y)

            plt.xlim([0.0, 1.0])
            plt.ylim([0.0, 1.05])
            plt.legend(legend[xlabel + ylabel], loc='upper right', prop=dict(size=12))
    plt.show()


def show_expected_value(Y_test, Y_prob, nclass, distribution, sample_weight=None, filename='temp', sys_show=False):
    threshold, precision, recall, _, _, accuracy = compute_precision_recall_volume_BV(Y_test, Y_prob, nclass,
                                                                                      distribution, sample_weight)
    precision_interval = compute_metrics_interval(metric=precision, nclass=nclass, n_samples=len(Y_test))
    recall_interval = compute_metrics_interval(metric=recall, nclass=nclass, n_samples=len(Y_test))
    #    LOGGER.info('shape of precision_interval', precision_interval.shape, precision_interval[:,0].shape)
    #    #interval = np.mean(precision_interval[:,0,0] -, axis =1)
    #    for (u,v) in zip(precision[:,0], precision_interval[:,0,1]):
    #        LOGGER.info(u,v)
    plt.figure()
    for c in range(nclass):
        # plt.subplot('21'+str(c+1))
        # plot_mean_and_CI(threshold,  precision[:,c], precision_interval[:,c,0], precision_interval[:,c,1])#,
        plot_mean_and_CI(threshold, recall[:, c], recall_interval[:, c, 0], recall_interval[:, c, 1])
    #        # Create blue bars
    ##        plt.bar(threshold, precision[:,c], color = 'blue', edgecolor = 'black',
    ##                yerr=interval, capsize=None, label='poacee')
    #
    #        break
    plt.xlabel('threshold', fontsize=14)
    plt.ylabel('recall', fontsize=14)
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.legend(['Recall for class 0', 'Recall for class 1'], loc='upper right', prop=dict(size=14))
    plt.title('Recall with 95% confidence interval')
    plt.show()


def plot_size_of_bins(Y_test, Y_prob, nclass, distribution, sample_weight=None, filename='temp'):
    '''
    Given a value of one metric (precsion, recall or volumne) or threshold, find other metrics
    '''
    threshold, precision, recall, volume, _ = compute_precision_recall_volume_BV(Y_test, Y_prob, nclass,
                                                                                 distribution, sample_weight)

    target_names = [str(i) for i in range(len(np.unique(Y_test)))]

    # Y_test = to_categorical(Y_test)
    threshold = np.array(range(0, 101, 5)) / 100.0
    TPs = np.zeros((len(threshold), nclass))
    TNs = np.zeros((len(threshold), nclass))
    FPs = np.zeros((len(threshold), nclass))
    FNs = np.zeros((len(threshold), nclass))
    business_values = np.zeros((len(threshold), nclass))

    for (i, T) in enumerate(threshold):
        # business_value = 0
        for c in range(nclass):
            Y_pred = (Y_prob[:, c] >= T).astype(np.int)
            # GE:[1] * len(threshold)})
    #
    #    df = pd.concat([df_male, df_female])
    df = pd.DataFrame({'threshold': threshold, 'tp': TPs[:, 0], 'tn': TNs[:, 0], 'fp': FPs[:, 0], 'fn': FNs[:, 0]})
    #    LOGGER.info(df.head())
    #    LOGGER.info(df.tail())
    # df.set_index(['threshold'], inplace=True)
    #LOGGER.info(df.head())
    # sns.barplot(x='threshold', y = [['tp', 'tn']], hue='gender', data=df)
    df = df.melt('threshold', var_name='metrics', value_name='count')
    # LOGGER.info(df.head())
    sns.barplot(x="threshold", y="count", hue='metrics', data=df)
    plt.xticks(rotation=45)
    plt.show()

def show_important_features(features_lgb, model, filename='temp.png', num_imp_feats=30, sys_show=False):
    sns.set(font_scale=1.5)
    plt.close()  # to reset all figures before, open new figure
    df_feature_importance = pd.DataFrame({'feature': features_lgb, 'importance': model.get_feature_importance()})
    best_features_lgb = df_feature_importance.sort_values(by="importance", ascending=False)[:num_imp_feats]
    sns.barplot(x="importance", y="feature", data=best_features_lgb)
    plt.title('LightGBM Features')
    fig = plt.gcf()
    fig.set_size_inches((18, 12), forward=False)
    fig.savefig(filename)

    if sys_show:
        plt.show()
