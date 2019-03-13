#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mar 12 17:46 2019

@author: phongdk
"""

import numpy as np
import pandas as pd


CURRENT_YEAR = 2019
BASE_DATE = pd.Timestamp('2019-03-10')       # compute age at the moment users access the Internet, not fix like this
LOW_AGE = 22
HIGH_AGE = 29


def get_target(x):
    if x < LOW_AGE or x > HIGH_AGE:
        return 0
    return 1


def load_data():
    fb_df = pd.read_csv(fb_filename, sep=' ', header=None, names=['raw_uid', 'gender', 'year'])
    fb_hash_id_df = pd.read_csv(fb_hash_id_filename, compression='gzip')
    df = pd.merge(fb_hash_id_df, fb_df, left_on='raw_uid', right_on='raw_uid', how='left')
    df.dropna(inplace=True)

    try:
        df['age'] = CURRENT_YEAR - df['year']
    except:
        df['year'] = pd.to_datetime(df['year'], format='%Y-%m-%d', errors='coerce')
        df = df[df['year'].notnull()]  # remove all rows that have incorrect year
        df['age'] = (BASE_DATE - df['year']).astype('<m8[Y]')
    df['age_group'] = df['age'].apply(lambda x: get_target(x))
    df = df[['user_id', 'age_group']]
    return df


if __name__ == '__main__':
    fb_hash_id_filename = 'external_data/facebook_hash_id.csv.gz'
    fb_filename = 'external_data/facebook.csv'
    df = load_data()
    df.to_csv('external_data/new_age_label_22_29.csv', index=False)
    # print(df.shape)
    # print(df.head(20))
