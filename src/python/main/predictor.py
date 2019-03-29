#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mar 12 17:47 2019

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
# from utils import normalize_topic_distribution
import warnings
from redisConnection import connectRedis, get_browser_id
warnings.filterwarnings("ignore")

OPTIMAL_THRESHOLD_FILENAME = 'Optimal_threshold.txt'
BALANCE_THRESHOLD = 0.5
GAP_INVENTORY = 0.05
NUM_THREADS = 16


def convert_hashID_to_browser_id(df):
    LOGGER.info("Shape before convert HashId {}".format(df.shape))
    r = connectRedis()
    df["browser_id"] = df["user_id"].apply(lambda x: get_browser_id(r, x))
    df.dropna(subset=["browser_id"], inplace=True)
    LOGGER.info("Shape before convert HashId {}".format(df.shape))
    return df


def prediction_stage(filename, path, target_label=1):
    LOGGER.info('-------------------- Load Test Data. ----------------------------')
    newest_model = sorted(os.listdir(directory), reverse=True)[0]       # find the newest model to predict
    lgb_models = sorted(glob.glob(os.path.join(path, newest_model, "*fold_[0-4].pkl")))

    LOGGER.info('\nMake predictions...\n')
    models = [pickle.load(open(m_name, 'rb')) for m_name in lgb_models]
    lgb_result = []
    list_userid = []
    for chunk in tqdm(pd.read_csv(filename, compression='gzip', chunksize=chunk_size, index_col='user_id',
                                  dtype={'user_id': str})):
        LOGGER.info(chunk.shape)
        list_userid.extend(list(chunk.index))
        if "gender" in chunk.columns:
            chunk.drop(columns=['gender', 'age_group'], inplace=True)

        chunk_result = [model.predict_proba(chunk, num_threads=NUM_THREADS) for model in models]  # prediction for each chunk
        chunk_result = np.array(chunk_result).mean(axis=0)

        lgb_result.extend(chunk_result)

    lgb_result = np.array(lgb_result, dtype=np.float16)

    if lgb_result.shape[1] == 2:  # binary classification
        with open(os.path.join(directory, newest_model, OPTIMAL_THRESHOLD_FILENAME), 'r') as f:
            optimal_threshold = float(f.read()) if is_best_threshold else BALANCE_THRESHOLD
            LOGGER.info("------------------Using threshold --------- : {}".format(optimal_threshold))
            final_result = np.array(lgb_result[:, 1] > optimal_threshold, dtype=np.int16)
    else:
        final_result = np.argmax(lgb_result, axis=1)

    df = pd.DataFrame({'user_id': list_userid, 'target': final_result})
    df = df[df['target'] == target_label]
    df = convert_hashID_to_browser_id(df)
    df['category_id'] = cate_id
    LOGGER.info(df.head())
    df[['browser_id', 'category_id']].to_csv(output_filename, compression='gzip', index=False, header=None, sep=' ')
    LOGGER.info(output_filename)

    if cate_id2:  # if wanna more inventory, using argmax instead of best_threshold
        #final_result = np.argmax(lgb_result, axis=1)
        final_result = np.array(lgb_result[:, 1] > optimal_threshold - GAP_INVENTORY, dtype=np.int16)
        df = pd.DataFrame({'user_id': list_userid, 'target': final_result})
        df = df[df['target'] == target_label]
        df = convert_hashID_to_browser_id(df)
        df['category_id'] = cate_id2
        LOGGER.info(df.head())
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

    args = vars(ap.parse_args())
    directory = args['model_directory']
    test_filename = args['test']
    chunk_size = args['chunk_size']
    is_best_threshold = args['best_threshold']
    cate_id = args['cate_id']
    cate_id2 = args['cate_id2']

    output_filename = args['output'] if args['output'].endswith('.gz') else '{}.csv.gz'.format(args['output'])

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

    prediction_stage(test_filename, directory)
