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
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import classification_report
import sys

sys.path.append('src/python/utils')
from utils import normalize_topic_distribution


def load_new_label(filename):
    print('-------------------LOAD NEW LABEL ------------------')
    df = pd.read_csv(filename, dtype={'user_id': str})
    df.drop_duplicates(subset=['user_id'], inplace=True)
    df.set_index('user_id', inplace=True)
    return df


def load_data(filename, nrows=None):
    print('-------------------LOAD TRAINING DATA ------------------')
    df = pd.read_csv(filename, compression='gzip', nrows=nrows, index_col='user_id',
                     dtype={'user_id': str, 'gender': int, 'age_group': int})
    return df


def change_label(df, x):
    try:
        return df.loc[str(x)].values[0]
    except:
        return None


def train_lgb(fold, x_train, y_train, x_valid, y_valid, lgb_path):
    params = {"objective": "binary", "metric": 'binary_logloss', "max_depth": 8, "min_child_samples": 20,
              "reg_alpha": 1, "reg_lambda": 1, "num_leaves": 257, "learning_rate": 0.01, "subsample": 0.8,
              "colsample_bytree": 0.8, "verbosity": -1, 'is_unbalance': True, "num_threads": 32, "n_estimators": 20000}
              #'num_class': len(np.unique(y_train))}

    model = lgb.LGBMClassifier(**params)
    model.fit(x_train, y_train, eval_set=(x_valid, y_valid), early_stopping_rounds=200, verbose=1000)
    cv_val = model.predict_proba(x_valid)
    # Save LightGBM Model
    pickle.dump(model, open(os.path.join(lgb_path, 'lgb_fold_{}.pkl').format(fold), 'wb'))
    return cv_val

    # lgb_params = {"objective": "multiclass", "metric": ['multi_logloss'], "max_depth": 8, "min_child_samples": 20,
    #               "reg_alpha": 1, "reg_lambda": 1, "num_leaves": 257, "learning_rate": 0.01, "subsample": 0.8,
    #               "colsample_bytree": 0.8, "verbosity": -1, 'is_unbalance': True, "num_threads": 32,
    #               'num_class': len(np.unique(y_train))}
    #
    # trn_data = lgb.Dataset(x_train, label=y_train)
    # val_data = lgb.Dataset(x_valid, label=y_valid)
    # num_round = 20000
    # model = lgb.train(lgb_params, trn_data, num_round, valid_sets=[val_data, trn_data], verbose_eval=1000,
    #                   early_stopping_rounds=100)
    # cv_val = model.predict(x_valid, model.best_iteration)
    # # Save LightGBM Model
    # pickle.dump(model, open(os.path.join(lgb_path, 'lgb_fold_{}_microsoft.pkl').format(fold), 'wb'))
    # return cv_val


def train():
    # print('-------------------LOAD TRAINING DATA ------------------')
    train_df = load_data(train_filename)
    new_label_df = load_new_label(new_label_filename)

    # train_df['age_group'] = train_df.index.apply(lambda x: change_label(new_label_df, x))
    train_df['age_group'] = [change_label(new_label_df, x) for x in list(train_df.index)]
    train_df.dropna(inplace=True)
    #train_df = normalize_topic_distribution(train_df)
    print(train_df.shape)

    Y = train_df['age_group']
    # Y = train_df['gender']
    X = train_df.drop(columns=['gender', 'age_group'])

    nclass = len(np.unique(Y))
    print(X.shape, Y.shape, nclass)
    print('Counting target')
    print(Y.value_counts())

    lgb_cv_result = np.zeros((X.shape[0], nclass))
    splits = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    for fold, (trn_idx, val_idx) in enumerate(splits.split(X, Y)):
        x_train, y_train = X.iloc[trn_idx], Y.iloc[trn_idx]
        x_valid, y_valid = X.iloc[val_idx], Y.iloc[val_idx]
        lgb_cv_result[val_idx] += train_lgb(fold, x_train, y_train, x_valid, y_valid, lgb_path=directory)

    lgb_cv_result = np.argmax(lgb_cv_result, axis=1)
    print(Y.values)
    print(lgb_cv_result)
    print('\n *************** LightGBM VAL EVALUATION: *******************')
    print(classification_report(Y, lgb_cv_result))


def prediction_stage(df_path, lgb_path):
    print('Load Test Data.')
    df = pd.read_csv(df_path, compression='gzip', index_col='user_id')
    Y = df['age_group']

    df.drop(columns=['gender', 'age_group'], inplace=True)
    print('\nShape of Test Data: {}'.format(df.shape))
    lgb_models = sorted(glob.glob(os.path.join(lgb_path, "*fold_[0-4].pkl")))
    lgb_result = []
    print('\nMake predictions...\n')

    for m_name in lgb_models:
        # Load LightGBM Model
        # print(m_name)
        model = pickle.load(open(m_name, 'rb'))
        lgb_result.append(model.predict_proba(df))
        # lgb_result.append(model.predict(df, model.best_iteration))
    lgb_result = np.array(lgb_result).mean(axis=0)
    lgb_result = np.argmax(lgb_result, axis=1)
    print('\n *************** LightGBM TEST EVALUATION: *******************')
    print(classification_report(Y,lgb_result))


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument("-f", "--train", required=True, help="path to training file")
    ap.add_argument("-q", "--test", required=True, help="path to testing file")

    # ap.add_argument("-m", "--method", required=True, help="name of method")
    ap.add_argument("-d", "--directory", required=True, help="path to model directory")
    ap.add_argument("-t", "--is_train", required=False, nargs='?', help="whether train or not", const=True, type=bool,
                    default=False)
    # ap.add_argument("-o", "--output", required=False, nargs='?', help="path to output file", const='', default='')
    # ap.add_argument("-l", "--log_file", required=False, help="path to log file")

    args = vars(ap.parse_args())

    train_filename = args['train']
    test_filename = args['test']
    # method = args['method']  # 0: Naive    1: RF   2:LGBM  3:DL
    directory = args['directory']
    is_train = args['is_train']

    # train_filename = sys.argv[1]
    new_label_filename = "external_data/new_age_label_22_29.csv"
    train()
    prediction_stage(test_filename, directory)
