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
from sklearn.metrics import classification_report, roc_auc_score, f1_score


import sys
sys.path.append('src/python/utils')
sys.path.append('src/python/model')
#from utils import normalize_topic_distribution
from LGBOptimizer import LGBOptimizer
import warnings
warnings.filterwarnings("ignore")

OPTIMAL_THRESHOLD_FILENAME = 'Optimal_threshold.txt'


def load_new_label(filename):
    LOGGER.info('-------------------LOAD NEW LABEL ------------------')
    df = pd.read_csv(filename, dtype={'user_id': str})
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


# def train_lgb(fold, x_train, y_train, x_valid, y_valid, lgb_path):
#     params = {"objective": "binary", "metric": 'auc', "max_depth": 8, "min_child_samples": 20,
#               "reg_alpha": 1, "reg_lambda": 1, "num_leaves": 13, "learning_rate": 0.01, "subsample": 0.8,
#               "colsample_bytree": 0.8, "verbosity": -1, 'is_unbalance': True, "num_threads": 32,
#               "num_iterations": 200000}
#     # 'num_class': len(np.unique(y_train))}
#
#     model = lgb.LGBMClassifier(**params)
#     model.fit(x_train, y_train, eval_set=(x_valid, y_valid), early_stopping_rounds=200, verbose=1000)
#     cv_val = model.predict_proba(x_valid)
#     # Save LightGBM Model
#     pickle.dump(model, open(os.path.join(lgb_path, 'lgb_fold_{}.pkl').format(fold), 'wb'))
#     return cv_val
#
#     # lgb_params = {"objective": "multiclass", "metric": ['multi_logloss'], "max_depth": 8, "min_child_samples": 20,
#     #               "reg_alpha": 1, "reg_lambda": 1, "num_leaves": 257, "learning_rate": 0.01, "subsample": 0.8,
#     #               "colsample_bytree": 0.8, "verbosity": -1, 'is_unbalance': True, "num_threads": 32,
#     #               'num_class': len(np.unique(y_train))}
#     #
#     # trn_data = lgb.Dataset(x_train, label=y_train)
#     # val_data = lgb.Dataset(x_valid, label=y_valid)
#     # num_round = 20000
#     # model = lgb.train(lgb_params, trn_data, num_round, valid_sets=[val_data, trn_data], verbose_eval=1000,
#     #                   early_stopping_rounds=100)
#     # cv_val = model.predict(x_valid, model.best_iteration)
#     # # Save LightGBM Model
#     # pickle.dump(model, open(os.path.join(lgb_path, 'lgb_fold_{}_microsoft.pkl').format(fold), 'wb'))
#     # return cv_val


# def train():
#     # LOGGER.info('-------------------LOAD TRAINING DATA ------------------')
#     train_df = load_data(train_filename)
#     new_label_df = load_new_label(new_label_filename)
#
#     # train_df['age_group'] = train_df.index.apply(lambda x: change_label(new_label_df, x))
#     train_df['age_group'] = [change_label(new_label_df, x) for x in list(train_df.index)]
#     train_df.dropna(inplace=True)
#     #train_df = normalize_topic_distribution(train_df)
#     LOGGER.info(train_df.shape)
#
#     Y = train_df['age_group']
#     # Y = train_df['gender']
#     X = train_df.drop(columns=['gender', 'age_group'])
#
#     nclass = len(np.unique(Y))
#     LOGGER.info(X.shape, Y.shape, nclass)
#     LOGGER.info('Counting target')
#     LOGGER.info(Y.value_counts())
#
#     lgb_cv_result = np.zeros((X.shape[0], nclass))
#     splits = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
#     for fold, (trn_idx, val_idx) in enumerate(splits.split(X, Y)):
#         x_train, y_train = X.iloc[trn_idx], Y.iloc[trn_idx]
#         x_valid, y_valid = X.iloc[val_idx], Y.iloc[val_idx]
#         lgb_cv_result[val_idx] += train_lgb(fold, x_train, y_train, x_valid, y_valid, lgb_path=directory)
#
#     lgb_cv_result = np.argmax(lgb_cv_result, axis=1)
#     LOGGER.info(Y.values)
#     LOGGER.info(lgb_cv_result)
#     LOGGER.info('\n *************** LightGBM VAL EVALUATION: *******************')
#     LOGGER.info(classification_report(Y, lgb_cv_result))

def get_optimal_binary_threshold(y, y_preds):
    LOGGER.info("-------------Finding optimal threshold ------------------------")
    best_score = 0
    best_threshold = 0
    for threshold in np.arange(0.2, 0.601, 0.01):
        score = f1_score(y, np.array(y_preds) > threshold)
        if best_score < score:
            best_score = score
            best_threshold = threshold
    LOGGER.info('best threshold is {:.4f} with F1 score: {:.4f}'.format(best_score, best_threshold))
    return best_threshold


def train():
    # LOGGER.info('-------------------LOAD TRAINING DATA ------------------')
    train_df = load_data(train_filename, nrows=None)
    new_label_df = load_new_label(new_label_filename)

    # train_df['age_group'] = train_df.index.apply(lambda x: change_label(new_label_df, x))
    train_df['age_group'] = [change_label(new_label_df, x) for x in list(train_df.index)]
    train_df.dropna(inplace=True)
    # train_df = normalize_topic_distribution(train_df)
    LOGGER.info(f"-------------TRAINING SET SHAPE : {train_df.shape}-------------------")
    train_df.drop(columns=['gender'], inplace=True)
    target_column = 'age_group'
    LOGGER.info('---------------------------Optimize parameter for LGBMs------------------------------')
    lgb_optimizer = LGBOptimizer(train_df, target_column=target_column, out_dir=hyper_params_path, n_jobs=32)
    if is_optimize:
        lgb_optimizer.optimize(metrics="roc_auc_score", n_splits=5, cv_type=StratifiedKFold, maxevals=200,
                               do_predict_proba=None)

    lgb_cv_result = lgb_optimizer.fit_data(path_save_model=directory, name='lgb')
    if len(train_df[target_column].unique()) == 2:  # binary classification
        optimal_threshold = get_optimal_binary_threshold(train_df[target_column], lgb_cv_result[:,1])
        final_result = np.array(lgb_cv_result[:, 1] > optimal_threshold, dtype=np.int16)
        with open(os.path.join(directory, OPTIMAL_THRESHOLD_FILENAME), 'w') as f:
            f.write(str(optimal_threshold))
    else:
        final_result = np.argmax(lgb_cv_result, axis=1)
    LOGGER.info('\n *************** LightGBM VAL EVALUATION: *******************')
    LOGGER.info(classification_report(train_df[target_column], final_result))
    LOGGER.info(classification_report(train_df[target_column], np.argmax(lgb_cv_result, axis=1)))


def prediction_stage(df_path, lgb_path):
    LOGGER.info('Load Test Data.')
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
    LOGGER.info(f"\n{classification_report(Y, final_result)}")
    LOGGER.info(f" AUC : {roc_auc_score(Y, final_result)} ")
    LOGGER.info(f"\n{classification_report(Y, np.argmax(lgb_result, axis=1))}")
    LOGGER.info(f" AUC : {roc_auc_score(Y, np.argmax(lgb_result, axis=1))} ")


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument("-f", "--train", required=True, help="path to training file")
    ap.add_argument("-q", "--test", required=True, help="path to testing file")
    ap.add_argument("-d", "--directory", required=True, help="path to model directory")
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
    ap.add_argument("-l", "--log_file", required=False, help="path to log file")

    args = vars(ap.parse_args())
    train_filename = args['train']
    test_filename = args['test']
    hyper_params_path = args['hyperparams']
    directory = args['directory']
    is_train = args['is_train']
    is_optimize = args['is_optimize']

    if not os.path.exists(directory):
        os.makedirs(directory)

    new_label_filename = "external_data/new_age_label_30_40.csv"

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
