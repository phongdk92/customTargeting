#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Fri Aug 31 12:06:00 2018

@author: phongdk
"""

import os
import logging
import numpy as np
import pandas as pd
import configparser
from sklearn.metrics import accuracy_score, confusion_matrix,  classification_report, precision_recall_curve, \
                    average_precision_score,recall_score, precision_score, f1_score
from sklearn.manifold import TSNE
from tensorflow.keras.utils import to_categorical
from visualization import plot_TPRV, plot_business_values

#from extract_fields_matching_URL import get_data_from_server
#def processURL(fields, date):
#    set_fields = ', ' .join(fields)
#    query = "SELECT {} from browser.clickdata \
#        WHERE event_date = '{}' AND request IN temp_url".format(set_fields, date)
#    external = "--external --file {} --name='temp_url' --structure='url String'".format(external_file)
#    output = get_data_from_server(query, external)
#    return output
path_figure = '/home/phongdk/data_ads_targeting/figure'
category_file = '/home/phongdk/data_ads_targeting/category_csv/'
active_user_file = '/home/phongdk/data_ads_targeting/uid_features/active_user.csv.gz'
age_group = [17,24,34,44,54]  # 0-17, 18-24, 25-34, 35-44, 45-54, 55+
#age_group = [14, 17, 24, 34, 44]#, 54]
current_year = 2018
basedate = pd.Timestamp('2018-08-30')       # compute age at the moment users access the Internet, not fix like this
SCALE = 100
CF = 1.0    #to compute Business Value

LOGGER = logging.getLogger(__name__)

if not os.path.exists(path_figure):
    os.mkdir(path_figure)

def age_to_age_group(age):
    for (i, ag) in enumerate(age_group):
        if age <= ag:
            return i
    return len(age_group)  #55+

def read_data(filename):
    df_cats = pd.read_csv(filename, sep = ' ', names = ['user_id', 'gender', 'year'])#, index_col='uid')
    df_cats = df_cats[df_cats['gender'] > 0]   # 0 means undefined
    df_cats['gender'] =  df_cats['gender'] - 1
    try:
        df_cats['age'] = current_year - df_cats['year']
    except:
        df_cats['year'] = pd.to_datetime(df_cats['year'], format='%Y-%m-%d', errors='coerce')
        df_cats = df_cats[df_cats['year'].notnull()]   #remove all rows that have incorrect year
        df_cats['age'] = (basedate- df_cats['year']).astype('<m8[Y]')
    return df_cats

def get_label_users(to_age_group=True):
    #df_cats = pd.concat([read_data(filename) for filename in ["cats.csv", "facebook.csv"]])
    df_cats = read_data(os.path.join(category_file, "facebook.csv"))
    if to_age_group:
        df_cats['age_group'] = df_cats['age'].apply(lambda x: age_to_age_group(x))
        df_cats.drop(columns=['year', 'age'], inplace=True)         #remove duplicates
    else:
        df_cats.rename(columns={'age':'age_group'}, inplace=True)
        df_cats.drop(columns=['year'], inplace=True)
    #LOGGER.info 'the number of uid who have labels for gender and age', len(df_cats)
    df_cats.drop_duplicates(subset=['user_id'], keep='first', inplace=True)
    LOGGER.info('the number of user_id who have labels for gender and age after removing duplicates {}'.format(len(df_cats)))
    df_cats.set_index(['user_id'], inplace=True)
    return df_cats

def split_train_test_hash(df):
    uids = df.index
    try:
        train_set_id = [x for x in uids if int(x) % 100 < 80 ]
    except:
        train_set_id = [x for x in uids if hash(x) % 100 < 80 ]
    test_set_id = list(set(uids) - set(train_set_id))
    assert len(train_set_id) + len(test_set_id) == len(uids)
    return train_set_id, test_set_id

def get_users_labels_matching(df_features, df_cats):     #filter active users that have labels
    LOGGER.info('matching users-labels')
    try:
        df_features.set_index('user_id', inplace=True)
    except:
        df_features.set_index('uid', inplace=True)
    LOGGER.info('the number of uid who have features {}'.format(len(df_features)))
    #df = pd.merge([df_features, df_cats], axis =1, join_axes=[df_features.index, df_cats.index])
    df = pd.merge(df_features, df_cats, left_on=df_features.index, right_on=df_cats.index, how='left')
    df = df.dropna()
    df.rename(columns={'key_0':'user_id'}, inplace=True)
    df.set_index('user_id', inplace=True)
    LOGGER.info('The number of users for training (intersection) {}'.format(len(df)))
    return df

def find_corresponding_pos_of_prob(threshold, p):
    for (pos, t) in enumerate(threshold):
        if (p < t):
            return pos   # t_{pos-1} < p < t_pos

def get_optimal_threshold(Y_test, Y_prob, distribution, specific_classes=None, sample_weight=None,
                          path_figure='.', filename='temp', sys_show=False, show_score=False):
    nclass = Y_prob.shape[1]
    specific_classes = sorted(specific_classes) if specific_classes else range(nclass)
    threshold, precision, recall, volume, business_values, _ = compute_precision_recall_volume_BV(Y_test, Y_prob,
                                                                                                  nclass, distribution,
                                                                                                  sample_weight)
    if sample_weight is None:
        sample_weight = np.ones(len(Y_test))
    optimal_threshold = -1
    optimal_segment_threshold = np.ones(nclass) * -1
    MIN_THRESHOLD_CLASS = 1.0 / nclass  # 1.0 / segment_count
    iteration = -1
    ylim_maxBV = 0
    while iteration < 100:
        LOGGER.info('****'*20 + "{}".format(iteration))
        iteration += 1

        total_BV = np.sum(business_values, axis=1)
        max_BV_pos = np.argmax(total_BV)
        max_BV = total_BV[max_BV_pos]
        ylim_maxBV = max(ylim_maxBV, max_BV + 0.2)
        plot_business_values(threshold, precision, recall, volume, business_values, specific_classes,
                               MIN_THRESHOLD_CLASS,
                               path_figure, filename.replace('.png', '_{}.png'.format(iteration)), sys_show,
                               show_score, ylim_maxBV=ylim_maxBV)

        # max_BV_pos_class = np.argmax(business_values, axis=0)  # max business value for each class
        # max_BV_class = np.max(business_values, axis=0)
        # threshold_class = threshold[max_BV_pos_class]

        segment_threshold = [max(MIN_THRESHOLD_CLASS, threshold[np.argmax(business_values[:, c])]) for c in
                             range(nclass)]
        LOGGER.info('---' * 15)
        LOGGER.info([np.argmax(business_values[:, c]) for c in range(nclass)])
        LOGGER.info('Optimal threshold for each class {}'.format(segment_threshold))
        LOGGER.info('Optimal threshold : {} \t , BV: {} '.format(optimal_threshold, max_BV))

        # if (optimal_threshold == threshold[max_BV_pos]):
        #     return optimal_threshold, max_BV, segment_threshold

        if np.allclose(optimal_segment_threshold, segment_threshold) and optimal_threshold == threshold[max_BV_pos]:
            return threshold[max_BV_pos], max_BV, segment_threshold
        optimal_segment_threshold = segment_threshold
        optimal_threshold = threshold[max_BV_pos]
        # start iteration
        cnt_class = np.zeros((len(threshold), nclass))
        total_cnt_class = np.zeros((len(threshold), nclass)) + 1e-6
        for k in range(len(Y_prob)):  # iterate all samples
            active_class = [i for (i, prob) in enumerate(Y_prob[k]) if prob >= segment_threshold[i]]
            corresponding_pos_for_segment = [find_corresponding_pos_of_prob(threshold, prob) if prob >= segment_threshold[i]
                                             else None for (i, prob) in enumerate(Y_prob[k])]

            for (idx, pos) in enumerate(corresponding_pos_for_segment):
                total_cnt_class[:pos, idx] += sample_weight[k]
            if len(active_class) >= 2:
                for c in active_class:
                    #for u in range(0, corresponding_pos_for_segment[c]):  # descrease BV of class c with threshold <= pc
                        u = corresponding_pos_for_segment[c]
                        flag = 0
                        for another_c in active_class:
                            if c != another_c:
                                BV_c = business_values[u, c]
                                BV_another_c = business_values[corresponding_pos_for_segment[another_c], another_c]
                                if (BV_c < BV_another_c) or (BV_c == BV_another_c and Y_prob[k, c] < Y_prob[k, another_c]):
                                    flag = 1
                                    break
                        cnt_class[:u, c] += sample_weight[k] * flag
        ratio_lost_volumn = cnt_class / total_cnt_class
        business_values = business_values * (1-ratio_lost_volumn)
    return threshold[max_BV_pos], max_BV, segment_threshold

def get_threshold_with_max_f1_score(Y_test, Y_prob, sample_weight=None):
    #threshold, _, _, _, _, _, f1score = compute_precision_recall_volume_BV(Y_test, Y_prob, distribution, sample_weight)
    Y_test = to_categorical(Y_test)
    threshold = np.array(range(0, 101, 2)) / 100.0
    # f1score = []
    # for (i, T) in enumerate(threshold):
    #     c = 1 #compute based on female
    #     Y_pred = (Y_prob[:, c] >= T).astype(np.int)
    #     f1score.append(f1_score(Y_test[:, c], Y_pred, sample_weight=sample_weight))
    # for (t, s) in zip(threshold, f1score):
    #     LOGGER.info(t, s)
    # pos = np.argmax(f1score)
    # T = threshold[pos]
    # return T
    nclass = Y_prob.shape[1]
    m = len(threshold)
    f1score = np.zeros((m, nclass))

    for (i, T) in enumerate(threshold):
        for c in range(nclass):
            Y_pred = (Y_prob[:, c] >= T).astype(np.int)
            f1score[i][c] = f1_score(Y_test[:, c], Y_pred, sample_weight=sample_weight)
    position_max_f1score = np.argmax(f1score, axis=0)
    threshold_each_class = threshold[position_max_f1score]
    LOGGER.info(position_max_f1score)
    LOGGER.info('threshold each class {}'.format(threshold_each_class))
    return threshold_each_class

def compute_precision_recall_volume_BV(Y_test, Y_prob, nclass, distribution, sample_weight=None):
    #X = classifier.prognoze(observation)
    #CDF(x) = P(X < x)user with weight w counte w times for P calculationsas CDF(x) = Sum(w * int(X < x)) / Sum(w)
    '''
    1.1 Is ratio of target in general set is the ratio of the target in global distribution or in test set?
    In my experiments, I understand it as the ﬁrst one?
    --> After reweighting ratio of the weighted target in test set has to be approximately same, as ratio of the target in global distribution
    --> ratio_of_target_in_general_set is similar to the proprotion of this taret in the global set
    '''
    LOGGER.info('----------------- Value of CF : {}'.format(CF))
    #target_names = [str(i) for i in range(len(np.unique(Y_test)))]

    Y_test = to_categorical(Y_test)
    threshold = np.array(range(0,101,2)) / 100.0
    m = len(threshold)
    volume = np.zeros((m, nclass))
    recall = np.zeros((m, nclass))
    precision = np.zeros((m, nclass))
    f1score = np.zeros((m, nclass))
    business_values = np.zeros((m, nclass))
    accuracy = np.zeros((m, nclass))

    for (i, T) in enumerate(threshold):
        for c in range(nclass):
            Y_pred = (Y_prob[:, c] >= T).astype(np.int)
            volume[i][c] = np.sum(Y_pred) * 1.0 / len(Y_test) if sample_weight is None else np.sum(Y_pred * sample_weight) / np.sum(sample_weight)
            recall[i][c]=  recall_score(Y_test[:, c], Y_pred, sample_weight= sample_weight)
            precision[i][c] = precision_score(Y_test[:,c], Y_pred, sample_weight= sample_weight)
            f1score[i][c] = f1_score(Y_test[:,c], Y_pred, sample_weight= sample_weight)
            accuracy[i][c] = accuracy_score(Y_test[:,c], Y_pred, sample_weight= None, normalize=False)
            business_values[i][c] = volume[i][c] * (CF *precision[i][c] - distribution[c])  /  distribution[c]
    return threshold, precision, recall, volume, business_values, accuracy

def compute_metrics_interval(metric, nclass, n_samples):
    LOGGER.info('compute_metric_interval')
    metric_interval = np.zeros((len(metric), nclass, 2))    #low and high
    for i in range(len(metric)):
        for c in range(nclass):
            #LOGGER.info(metric[i][c])
            metric_interval[i][c] = compute_wilson_score_interval(metric[i][c], n_samples)
    return metric_interval
    #recall_interval = np.zeros((m, nclass, 2))       #low and high
    #precision_interval[i][c] = compute_wilson_score_interval(precision[i][c], len(Y_test))   # take care of weighted test, is it involved in compute interval
    #recall_interval[i][c] = compute_wilson_score_interval(recall[i][c], len(Y_test))

#compute wilson score interval for metrics
def compute_wilson_score_interval(p_hat, n):
    #p_hat = n_success / n
    z = 1.96  # corresponds to 95% confidence
    p_low = ((p_hat + z*z/(2*n) - z * np.sqrt((p_hat*(1-p_hat)+z*z/(4*n))/n))/(1+z*z/n))
    p_high = ((p_hat + z*z/(2*n) + z * np.sqrt((p_hat*(1-p_hat)+z*z/(4*n))/n))/(1+z*z/n))
    p_low = max(0, p_low)
    p_high = min(1, p_high)
    return [p_low, p_high]

def retrieve_artifacts(df, prob, T_low, T_high):
    ''' Get artifacts in range [T_low, T_high]
        prob: probability of each sample according to a class we recognize anomally
        df: data frame with all information of samples
    '''
    LOGGER.info(f'Retrieve artifact with threshold from {T_low} and {T_high}')
    df['prob'] = prob
    df['mask'] = (prob < T_high ) & (prob > T_low)
    artifacts = df[df['mask']]
    LOGGER.info(len(artifacts))
    #LOGGER.info(artifacts['prob'].head())
    LOGGER.info(artifacts.head())

def load_data(filename, index='user_id', nrows=None):
    #df = cPickle.load(open(filename, 'rb'))
    df = pd.read_csv(filename, index_col=index, nrows=nrows)
    if 'gender' in df.columns:
        df['gender'] = df['gender'].astype('int8')
    if 'age_group' in df.columns:
        df['age_group'] = df['age_group'].astype('int8')
    return df

def get_features_and_labels(df):
    Y_gender= np.array(df['gender'].values)
    Y_age = np.array(df['age_group'].values)
    #df.drop(columns={'gender', 'age_group'}, inplace=True)
    #X = df.values
    X = np.array(df.drop(columns={'gender', 'age_group'}).values)
    return X, Y_gender, Y_age

def get_sample_weight(df, column, global_dist):
    ncount = np.array(df.groupby(column).size().values, dtype=float)
    category_weight = float(len(df)) / ncount * global_dist
    category_weight = category_weight / np.sum(category_weight)
    class_weight = {}
    for (i, v) in enumerate(category_weight):
        class_weight[i] = v
    '''RF: 1.0/v , LGBM: v'''
    weight = df.apply(lambda x: category_weight[int(x[column])], axis =1 )
    sample_weight = np.array(weight.values.tolist()) #/ SCALE
    return class_weight, sample_weight

def show_metric_scores(Y_test, Y_prob, nclass, distribution, specific_classes = None, sample_weight=None,
                       path_figure = '.', filename='temp', sys_show=False, show_score = False):
    #X = classifier.prognoze(observation)
    #CDF(x) = P(X < x)user with weight w counte w times for P calculationsas CDF(x) = Sum(w * int(X < x)) / Sum(w)
    '''
    1.1 Is ratio of target in general set is the ratio of the target in global distribution or in test set?
    In my experiments, I understand it as the ﬁrst one?
    --> After reweighting ratio of the weighted target in test set has to be approximately same, as ratio of the target in global distribution
    --> ratio_of_target_in_general_set is similar to the proprotion of this taret in the global set
    '''
    specific_classes = sorted(specific_classes) if specific_classes else range(nclass)
    threshold, precision, recall, volume, business_values, _ = compute_precision_recall_volume_BV(Y_test, Y_prob,
                                                                                                  distribution,
                                                                                                  sample_weight)
    MIN_THRESHOLD_CLASS = 1.0 / nclass  # 1.0 / segment_count
    return plot_business_values(threshold, precision, recall, volume, business_values, specific_classes,
                                  MIN_THRESHOLD_CLASS,
                                    path_figure, filename, sys_show, show_score)

def show_TPRV(Y_test, Y_prob, nclass, distribution, sample_weight = None, known_type = 'threshold',
             known_value=0.4, known_class=0,  filename='temp'):
    '''
    Given a value of one metric (precsion, recall or volumne) or threshold, find other metrics
    '''
    nclass = Y_prob.shape[1]
    threshold, precision, recall, volume, _, _ = compute_precision_recall_volume_BV(Y_test, Y_prob, distribution, sample_weight)

    plot_TPRV(threshold, precision, recall, volume, nclass, known_type=known_type, known_value=known_value,
              known_class=known_class, filename=filename)

def normalize_topic_distribution(df):
    LOGGER.info('normalize topic distribution')
    # for col in df.columns:#[1:257]:
    #     if 'td_t' in col:
    active_rows = [i for (i, x) in enumerate(df['td_read_doc'].values) if x > 0]    # avoid divide by 0
    for i in range(0, 256):
        col = 'td_t_{}'.format(i)
        df[col].iloc[active_rows] = df[col].iloc[active_rows] / df['td_read_doc'].iloc[active_rows]
    df.dropna(inplace=True)
    return df

def normalize_affinity(df):
    LOGGER.info('normalize affinity')
    active_rows = [i for (i, x) in enumerate(df['aff_num_g'].values) if x > 0]      # avoid divide by 0
    for col in ['aff_mal', 'aff_fem']:
        df[col].iloc[active_rows] = df[col].iloc[active_rows] / df['aff_num_g'].iloc[active_rows]

    active_rows = [i for (i, x) in enumerate(df['aff_num_a'].values) if x > 0]      # avoid divide by 0
    for i in range(10, 86):
        col = 'aff_a_{}'.format(i)
        df[col].iloc[active_rows] = df[col].iloc[active_rows] / df['aff_num_a'].iloc[active_rows]
    return df

def filter_out_shadows(Y_prob, thresholds):
    #TO DO: filter out samples that are shadow (predicted probability <= Threshold_class for each class)
    mask = np.sum((Y_prob > thresholds).astype('int'), axis=1)
    shadows = [i for (i, x) in enumerate(mask) if x == 0]
    non_shadows = [i for (i, x) in enumerate(mask) if x != 0]   #set(list(range(len(Y_prob)))) - set(shadows)
    LOGGER.info('Length of shadow: {}'.format(len(shadows)))
    LOGGER.info('Length of non-shadow: '.format(len(non_shadows)))
    # for s in shadows:
    #     LOGGER.info(Y_prob[s], thresholds)
    return np.array(shadows), np.array(non_shadows)

def make_directories(root):
    if not os.path.exists(root):
        os.makedirs(root)
    for folder in ['model', 'figure', 'segment_thresholds']:
        if not os.path.exists(os.path.join(root, folder)):
            os.mkdir(os.path.join(root, folder))

def write_configuration(file_config, gender_model, age_model, gender_thresholds, age_thresholds,
                        gender_f1score, age_f1score):
    config = configparser.ConfigParser()
    config['MODEL'] = {'Gender': gender_model,
                       'Age': age_model}
    config['THRESHOLD'] = {'Gender': ' '.join(map(str, gender_thresholds)),
                           'Age': ' '.join(map(str, age_thresholds))}
    config['F1_THRESHOLD'] = {'Gender': ' '.join(map(str, gender_f1score)),
                              'Age': ' '.join(map(str, age_f1score))}
    with open(file_config, 'w') as configfile:
        config.write(configfile)

def load_configuration(path_model):
    config = configparser.ConfigParser()
    config.read(os.path.join(path_model, 'config'))
    gender_model = os.path.join(path_model, config['MODEL']['Gender'])
    age_model = os.path.join(path_model, config['MODEL']['Age'])
    gender_threshold = np.array([float(x) for x in config['THRESHOLD']['Gender'].split()])
    age_threshold = np.array([float(x) for x in config['THRESHOLD']['Age'].split()])
    # gender_f1score_thresholds = np.array([float(x) for x in config['F1_THRESHOLD']['Gender'].split()])
    # age_f1score_thresholds = np.array([float(x) for x in config['F1_THRESHOLD']['Age'].split()])
    return gender_model, age_model, gender_threshold, age_threshold#, gender_f1score_thresholds, age_f1score_thresholds