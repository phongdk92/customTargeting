#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Mar 11 22:08:13 2019

@author: phongdk
"""

import numpy as np
import pandas as pd
import pickle
import json
import lightgbm as lgb
import os
import logging
from hyperparameter_hunter import Environment, CVExperiment, BayesianOptimization, Integer, Real, Categorical
from hyperparameter_hunter import optimization as opt
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import f1_score, roc_auc_score
import warnings
warnings.filterwarnings("ignore")

LOGGER = logging.getLogger(__name__)


class LGBOptimizer(object):
    def __init__(self, df, target_column='target', out_dir='HyperparameterHunterAssets', n_jobs=-1,
                 open_file=open):
        """
        Hyper Parameter optimization
        Comments: Hyperparameter_hunter (hereafter HH) is a fantastic package
        (https://github.com/HunterMcGushion/hyperparameter_hunter) to avoid
        wasting time as you optimise parameters. In the words of his author:
        "For so long, hyperparameter optimization has been such a time
        consuming process that just pointed you in a direction for further
        optimization, then you basically had to start over".
        Parameters:
        -----------
        trainDataset: FeatureTools object
            The result of running FeatureTools().fit()
        out_dir: Str
            Path to the output directory
        """

        self.data = df
        self.target_column = target_column
        self.n_jobs = n_jobs
        self.categorical_columns = None
        self.results_path = str(out_dir)
        self.OPEN_METHOD = open_file

    def optimize(self, metrics='f1_score', n_splits=5, cv_type=StratifiedKFold, maxevals=200, do_predict_proba=None):
        params = self.hyperparameter_space()
        extra_params = self.extra_setup()
        env = Environment(
            train_dataset=self.data,
            results_path=self.results_path,
            target_column=self.target_column,
            metrics=[metrics],
            do_predict_proba=do_predict_proba,
            cv_type=cv_type,
            cv_params=dict(n_splits=n_splits, shuffle=True, random_state=42),
        )
        # optimizer = opt.GradientBoostedRegressionTreeOptimization(iterations=maxevals)
        optimizer = opt.BayesianOptimization(iterations=maxevals)
        optimizer.set_experiment_guidelines(
            model_initializer=lgb.LGBMClassifier,
            model_init_params=params,
            model_extra_params=extra_params
        )
        LOGGER.info('------------------Optimizer Go--------------------')
        optimizer.go()
        # there are a few fixes on its way and the next few lines will soon be
        # one. At the moment, to access to the best parameters one has to read
        # from disc and access them
        LOGGER.info('Got best parameters')
        LOGGER.info(optimizer.best_experiment)

    def get_best_params(self, best_experiment_id=None):
        LOGGER.info('------------------Get best parameters-------------------')
        leaderboards = os.path.join(self.results_path, 'HyperparameterHunterAssets/Leaderboards/GlobalLeaderboard.csv')
        best_experiment_id = pd.read_csv(leaderboards, nrows=2, usecols=['experiment_id'],
                                         dtype={'experiment_id': str})['experiment_id'].values[0]
        best_experiment = os.path.join(self.results_path, 'HyperparameterHunterAssets/Experiments/Descriptions/',
                                       '{}.json'.format(best_experiment_id))
        with open(best_experiment) as best:
            best = json.loads(best.read())['hyperparameters']['model_init_params']
        return best

    def fit_data(self, path_save_model, name):
        X = self.data.drop(columns=[self.target_column])
        Y = self.data[self.target_column]

        best_params = self.get_best_params()
        LOGGER.info(best_params)
        nclass = len(Y.unique())
        lgb_cv_result = np.zeros((X.shape[0], nclass))
        splits = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

        for fold, (trn_idx, val_idx) in enumerate(splits.split(X, Y)):
            x_train, y_train = X.iloc[trn_idx], Y.iloc[trn_idx]
            x_valid, y_valid = X.iloc[val_idx], Y.iloc[val_idx]

            lgb_cv_result[val_idx] += self.__train_lgb(best_params, fold, x_train, y_train, x_valid, y_valid,
                                                       path_save_model, name)
        return lgb_cv_result

    def __train_lgb(self, best_params, fold, x_train, y_train, x_valid, y_valid, lgb_path, name):
        LOGGER.info(f'--------------- Training LGBM with FOLD ---------------------: {fold}')
        # model = pickle.load(self.OPEN_METHOD(os.path.join(lgb_path, f'{name}_fold_{fold}.pkl'), 'rb'))
        model = lgb.LGBMClassifier(**best_params)
        model.fit(x_train, y_train,
                  eval_set=[(x_valid, y_valid)],
                  verbose=1000,
                  early_stopping_rounds=300)
        pickle.dump(model, self.OPEN_METHOD(os.path.join(lgb_path, f'{name}_fold_{fold}.pkl'), 'wb'))
        cv_val = model.predict_proba(x_valid)
        return cv_val

    def hyperparameter_space(self, param_space=None):
        space = dict(
            objective="binary",
            boosting_type="gbdt",
            metric="binary_logloss",
            is_unbalance=True,
            boost_from_average=False,
            num_threads=self.n_jobs,
            learning_rate=Real(0.005, 0.01),
            # learning_rate=Categorical([0.005, 0.01]),
            num_leaves=Integer(8, 20),
            max_depth=-1,
            feature_fraction=0.041,
            bagging_freq=5,
            bagging_fraction=0.331,
            min_data_in_leaf=Categorical(range(42, 90, 10)),
            min_sum_hessian_in_leaf=10.0,
            num_iterations=200000)

        nclass = len(self.data[self.target_column].unique())
        if nclass > 2:
            LOGGER.info("*********Since the number classes > 2: so the objective function is 'multiclass' *********")
            space['objective'] = 'multiclass'
            space['metric'] = 'multi_logloss'
            space['num_class'] = nclass
            
        LOGGER.info("------------------------ Objective : {}--------------".format(space['objective']))

        if param_space:
            return param_space
        else:
            return space

    def extra_setup(self, extra_setup=None):

        extra_params = dict(
            early_stopping_rounds=300)
        # feature_name=self.colnames)
        # categorical_feature=self.categorical_columns)

        if extra_setup:
            return extra_setup
        else:
            return extra_params
