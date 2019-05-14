#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mar 12 17:46 2019

@author: phongdk
"""
import os
import glob
import numpy as np
import pandas as pd
import lightgbm as lgb
import argparse
import pickle
import logging
from datetime import datetime
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import classification_report, roc_auc_score, f1_score, precision_score, \
    confusion_matrix, recall_score, accuracy_score


import sys
sys.path.append('src/python/utils')
sys.path.append('src/python/model')
#from utils import normalize_topic_distribution
from LGBOptimizer import LGBOptimizer
import warnings
warnings.filterwarnings("ignore")

metrics_dict = dict(f1_score=f1_score, precision=precision_score, recall=recall_score, accuracy=accuracy_score)
OPTIMAL_THRESHOLD_FILENAME = 'Optimal_threshold.txt'
N_JOBS = 32


def load_new_label(filename):
    LOGGER.info('-------------------LOAD NEW LABEL ------------------')
    try:
        df = pd.read_csv(filename, dtype={'user_id': str})
    except:
        df = pd.read_csv(filename, compression='gzip', dtype={'user_id': str})
    df.drop_duplicates(subset=['user_id'], inplace=True)
    df.set_index('user_id', inplace=True)
    return df


def load_data(filename, nrows=None):
    LOGGER.info('-------------------LOAD DATA ------------------')
    df = pd.read_csv(filename, compression='gzip', nrows=nrows, index_col='user_id',
                     dtype={'user_id': str, 'gender': int, 'age_group': int})
    return df


def change_label(df, x):
    try:
        return df.loc[str(x)].values[0]
    except:
        return None


def get_optimal_binary_threshold(y, y_preds, metrics=f1_score):
    LOGGER.info("-------------Finding optimal threshold ------------------------")
    best_score = 0
    best_threshold = 0
    for threshold in np.arange(0.2, 0.721, 0.01):
        score = metrics(y, np.array(y_preds) > threshold)
        if best_score < score:
            best_score = score
            best_threshold = threshold
    LOGGER.info('best threshold is \t {:.4f} with {}: \t{:.4f}'.format(best_threshold, metrics.__name__, best_score ))
    return best_threshold


def train():
    # LOGGER.info('-------------------LOAD TRAINING DATA ------------------')
    train_df = load_data(train_filename, nrows=None)
    new_label_df = load_new_label(new_label_filename)

    # train_df['age_group'] = train_df.index.apply(lambda x: change_label(new_label_df, x))
    train_df['age_group'] = [change_label(new_label_df, x) for x in list(train_df.index)]
    train_df.dropna(inplace=True)
    # train_df = normalize_topic_distribution(train_df)
    LOGGER.info("-------------TRAINING SET SHAPE : {} -------------------".format(train_df.shape))
    train_df.drop(columns=['gender'], inplace=True)
    target_column = 'age_group'
    LOGGER.info('---------------------------Optimize parameter for LGBMs------------------------------')
    lgb_optimizer = LGBOptimizer(train_df, target_column=target_column, out_dir=hyper_params_path, n_jobs=N_JOBS)
    if is_optimize:
        lgb_optimizer.optimize(metrics="roc_auc_score", n_splits=5, cv_type=StratifiedKFold, maxevals=200,
                               do_predict_proba=None)

    lgb_cv_result = lgb_optimizer.fit_data(path_save_model=directory, name='lgb')
    if len(train_df[target_column].unique()) == 2:  # binary classification
        optimal_threshold = get_optimal_binary_threshold(train_df[target_column], lgb_cv_result[:, 1],
                                                         metrics=metric_optimization)
        final_result = np.array(lgb_cv_result[:, 1] > optimal_threshold, dtype=np.int16)
        with open(os.path.join(directory, OPTIMAL_THRESHOLD_FILENAME), 'w') as f:
            f.write(str(optimal_threshold))
    else:
        final_result = np.argmax(lgb_cv_result, axis=1)
    LOGGER.info('\n *************** LightGBM VAL EVALUATION: *******************')
    LOGGER.info(classification_report(train_df[target_column], final_result))
    LOGGER.info(classification_report(train_df[target_column], np.argmax(lgb_cv_result, axis=1)))


def prediction_stage(df_path, lgb_path):
    LOGGER.info('---------------Load Test Data.-----------------------------------')
    df = load_data(df_path)
    new_label_df = load_new_label(new_label_filename)
    df['age_group'] = [change_label(new_label_df, x) for x in list(df.index)]
    LOGGER.info(df.shape)
    df.dropna(inplace=True)
    LOGGER.info(df.shape)
    Y = df['age_group']

    df.drop(columns=['gender', 'age_group'], inplace=True)
    LOGGER.info('\nShape of Test Data: {}'.format(df.shape))
    lgb_models = sorted(glob.glob(os.path.join(lgb_path, "*fold_[0-4].pkl")))
    lgb_result = []
    LOGGER.info('\nMake predictions...\n')

    for m_name in lgb_models:
        # Load LightGBM Model
        # LOGGER.info(m_name)
        model = pickle.load(open(m_name, 'rb'))
        lgb_result.append(model.predict_proba(df))
        # lgb_result.append(model.predict(df, model.best_iteration))
    lgb_result = np.array(lgb_result).mean(axis=0)

    if lgb_result.shape[1] == 2:    # binary classification
        with open(os.path.join(directory, OPTIMAL_THRESHOLD_FILENAME), 'r') as f:
            optimal_threshold = float(f.read())
            final_result = np.array(lgb_result[:, 1] > optimal_threshold, dtype=np.int16)
    else:
        final_result = np.argmax(lgb_result, axis=1)
    LOGGER.info('\n *************** LightGBM TEST EVALUATION: *******************')
    LOGGER.info(classification_report(Y, final_result))
    LOGGER.info(" AUC : {} ".format(roc_auc_score(Y, final_result)))
    LOGGER.info(" Confusion matrix : \n {}".format(confusion_matrix(Y, final_result)))
    # LOGGER.info(f"\n{classification_report(Y, np.argmax(lgb_result, axis=1))}")
    # LOGGER.info(f" AUC : {roc_auc_score(Y, np.argmax(lgb_result, axis=1))} ")


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument("-f", "--train", required=True, help="path to training file")
    ap.add_argument("-q", "--test", required=True, help="path to testing file")
    ap.add_argument("-d", "--directory", required=True, help="path to model directory")
    ap.add_argument("-nl", "--new_label", required=True, help="path to new label file")

    '''
    Parameter (-op, -hp), -t should go together
    optimize --> output to hp --> train
    OR:
    train without optimize
    '''
    ap.add_argument("-hp", "--hyperparams", required=True, help="path to hyper-parameters directory")
    ap.add_argument("-t", "--is_train", required=False, nargs='?', help="whether train or not", const=True, type=bool,
                    default=False)
    ap.add_argument("-op", "--is_optimize", required=False, nargs='?', help="whether optimize parameters or not",
                    const=True, type=bool, default=False)
    ap.add_argument("-me", "--metric", required=False, help="metric to optimize threshold", type=str, default="f1_score")
    ap.add_argument("-l", "--log_file", required=False, help="path to log file")

    args = vars(ap.parse_args())
    train_filename = args['train']
    test_filename = args['test']
    directory = args['directory']
    new_label_filename = args['new_label']  # "external_data/new_age_label_30_40.csv"

    hyper_params_path = args['hyperparams']
    is_train = args['is_train']
    is_optimize = args['is_optimize']
    metric_optimization = metrics_dict[args['metric']]

    if not os.path.exists(directory):
        os.makedirs(directory)

    if args['log_file'] is not None:
        log_filename = args['log_file']
    else:
        LOG_DIR = "logs"
        if not os.path.exists(LOG_DIR):
            os.mkdir(LOG_DIR)
        log_filename = os.path.join(LOG_DIR, datetime.today().strftime("%Y-%m-%d.log"))

    logging.basicConfig(level=logging.INFO, filename=log_filename)
    LOGGER = logging.getLogger("main")

    LOGGER.info(args)
    LOGGER.info('----' * 20 + "{}".format(datetime.today()))
    if is_train:
        train()
    prediction_stage(test_filename, directory)
