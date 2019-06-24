#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Apr 16 17:56 2019

@author: phongdk
"""


import os
import glob
import numpy as np
import pandas as pd
import argparse
import pickle
import sys
from tqdm import tqdm
from datetime import datetime
import logging

sys.path.append('src/python/utils')
sys.path.append('src/python/model')
sys.path.append('src/python/db')
sys.path.append('src/python/hdfs_config')
# from utils import normalize_topic_distribution
from redisConnection import connectRedis, get_browser_id
from hdfs_config import storage_options, hdfs, HDFS_PREFIX
from distributed import Client
import dask.dataframe as dd
import dask
dask.config.set(hdfs_driver='hdfs3')  # so critical

import warnings
warnings.filterwarnings("ignore")

OPTIMAL_THRESHOLD_FILENAME = 'Optimal_threshold.pkl'
BALANCE_THRESHOLD = 0.5
GAP_INVENTORY = 0.05
NUM_THREADS = 24


def convert_hashID_to_browser_id(df):
    LOGGER.info("Shape before convert HashId {}".format(df.shape))
    r = connectRedis()
    df["browser_id"] = df["user_id"].apply(lambda x: get_browser_id(r, x))
    df.dropna(subset=["browser_id"], inplace=True)
    LOGGER.info("Shape before convert HashId {}".format(df.shape))
    return df


def load_models(path):
    LOGGER.info('-------------------- Load Model ----------------------------')
    if is_hdfs:
        newest_model = os.path.basename(sorted(hdfs.ls(path), reverse=True)[0])  # find the newest model to predict
        model_names = sorted(hdfs.glob(os.path.join(path, newest_model, "*fold_[0-4].pkl")))
    else:
        newest_model = sorted(os.listdir(directory), reverse=True)[0]  # find the newest model to predict
        model_names = sorted(glob.glob(os.path.join(path, newest_model, "*fold_[0-4].pkl")))
    LOGGER.info("The Newest model {}".format(newest_model))
    models = [pickle.load(OPEN_METHOD(m_name, 'rb')) for m_name in model_names]
    return models, newest_model


def get_predicted_values(df, models):
    result = [model.predict_proba(df, num_threads=NUM_THREADS) for model in models[0:1]]
    result = np.array(result).mean(axis=0)
    return result


def prediction_stage(filename, path, target_label=1):
    LOGGER.info('-------------------- Load Test Data. ----------------------------')
    models, newest_model = load_models(path)
    lgb_result = []
    list_userid = []
    print('Number of models : {}'.format(len(models)))
    assert len(models) > 0
    if is_hdfs:
        demographic_filenames = sorted(hdfs.glob(os.path.join(filename)))
        LOGGER.info("Number files to predicts : {}".format(len(demographic_filenames)))
        for (i, demo_filename) in enumerate(demographic_filenames):
            df = dd.read_csv(HDFS_PREFIX + demo_filename, dtype={'user_id': str}, compression='gzip', blocksize=None,
                             storage_options=storage_options).set_index("user_id").compute()
            list_userid.extend(list(df.index))
            print('Number of samples : -------------------- ', len(list_userid))
            LOGGER.info(demo_filename)
            LOGGER.info("Number of samples : {}".format(len(list_userid)))

            if "gender" in df.columns:
                df = df.drop(['gender', 'age_group'], axis=1)

            file_result = get_predicted_values(df, models)
            lgb_result.extend(file_result)
    else:   # prediction for each chunk
        for chunk in tqdm(pd.read_csv(filename, compression='gzip', chunksize=chunk_size, index_col='user_id',
                                      dtype={'user_id': str})):
            LOGGER.info(chunk.shape)
            list_userid.extend(list(chunk.index))
            if "gender" in chunk.columns:
                chunk.drop(columns=['gender', 'age_group'], inplace=True)

            chunk_result = [model.predict_proba(chunk, num_threads=NUM_THREADS) for model in models]
            chunk_result = np.array(chunk_result).mean(axis=0)

            lgb_result.extend(chunk_result)

    lgb_result = np.array(lgb_result, dtype=np.float16)

    if lgb_result.shape[1] == 2:  # binary classification
        optimal_threshold = pickle.load(OPEN_METHOD(os.path.join(directory, newest_model,
                                                                 OPTIMAL_THRESHOLD_FILENAME), 'rb')) \
            if is_best_threshold else BALANCE_THRESHOLD
        LOGGER.info("------------------Using threshold --------- : {}".format(optimal_threshold))
        final_result = np.array(lgb_result[:, 1] > optimal_threshold, dtype=np.int16)
    else:
        final_result = np.argmax(lgb_result, axis=1)

    df = pd.DataFrame({'user_id': list_userid, 'target': final_result})
    df = df[df['target'] == target_label]
    df = convert_hashID_to_browser_id(df)
    df['category_id'] = cate_id
    LOGGER.info(df.head())
    if is_hdfs:
        data = dd.from_pandas(df[['browser_id', 'category_id']], npartitions=1)
        data.to_csv([output_filename], compression='gzip', index=False, header=None, sep=' ',
                    storage_options=storage_options)
    else:
        df[['browser_id', 'category_id']].to_csv(output_filename, compression='gzip', index=False, header=None, sep=' ')
    LOGGER.info(output_filename)

    if cate_id2:  # if wanna more inventory, using argmax instead of best_threshold
        #final_result = np.argmax(lgb_result,axis=1)
        final_result = np.array(lgb_result[:, 1] > optimal_threshold - GAP_INVENTORY, dtype=np.int16)
        df = pd.DataFrame({'user_id': list_userid, 'target': final_result})
        df = df[df['target'] == target_label]
        df = convert_hashID_to_browser_id(df)
        df['category_id'] = cate_id2
        LOGGER.info(df.head())
        if is_hdfs:
            data = dd.from_pandas(df[['browser_id', 'category_id']], npartitions=1)
            data.to_csv([output_filename.replace("_75", "_50")], compression='gzip', index=False, header=None, sep=' ',
                        storage_options=storage_options)
        else:
            df[['browser_id', 'category_id']].to_csv(output_filename.replace("_75", "_50"), compression='gzip',
                                                     index=False, header=None, sep=' ')
        LOGGER.info(output_filename)


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument("-md", "--model_directory", required=True, help="path to model directory")
    ap.add_argument("-q", "--test", required=True, help="path to testing file")
    ap.add_argument("-o", "--output", required=True, help="path to output file")
    ap.add_argument("-cs", "--chunk_size", required=False, nargs='?',
                    help="chunk size for reading and processing a large file", type=int, default=-1)
    ap.add_argument("-l", "--log_file", required=False, help="path to log file")
    ap.add_argument("-bt", "--best_threshold", required=False, help="path to log file", type=bool, default=True)
    ap.add_argument("-cid", "--cate_id", required=True, help="Category ID", type=int)
    ap.add_argument("-cid2", "--cate_id2", required=False,
                    help="Category ID 2 (if wanna more inventory)", type=int)
    ap.add_argument("-hdfs", "--hdfs", required=False, nargs='?', help="metric to optimize threshold", type=bool,
                    const=True, default=False)

    args = vars(ap.parse_args())
    directory = args['model_directory']
    test_filename = args['test']
    chunk_size = args['chunk_size']
    is_best_threshold = args['best_threshold']
    cate_id = args['cate_id']
    cate_id2 = args['cate_id2']
    is_hdfs = args['hdfs']

    output_filename = args['output'] if args['output'].endswith('.gz') else '{}.gz'.format(args['output'])
    if args['log_file'] is not None:
        log_filename = args['log_file']
    else:
        LOG_DIR = "logs"
        if not os.path.exists(LOG_DIR):
            os.mkdir(LOG_DIR)
        log_filename = os.path.join(LOG_DIR, datetime.today().strftime("%Y-%m-%d.log"))

    logging.basicConfig(level=logging.INFO, filename=log_filename)
    LOGGER = logging.getLogger("main")

    LOGGER.info('----' * 20 + "{}".format(datetime.today()))
    LOGGER.info(args)

    client = Client('ads-target1v.dev.itim.vn:8786')
    print(client)

    if is_hdfs:
        if not hdfs.isdir(os.path.dirname(output_filename)):  # make directory on HDFS
            hdfs.makedirs(os.path.dirname(output_filename))
    else:
        if not os.path.isdir(os.path.dirname(output_filename)):
            os.makedirs(os.path.dirname(output_filename))

    OPEN_METHOD = hdfs.open if is_hdfs else open  # define open file method

    prediction_stage(test_filename, directory)
