#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mar 12 17:46 2019

@author: phongdk
"""

import numpy as np
import pandas as pd
import sys

CURRENT_YEAR = 2019
BASE_DATE = pd.Timestamp('2019-03-15')       # compute age at the moment users access the Internet, not fix like this
AGE_GROUP = [17, 29, 40, 54]  # 0-17, 18-29, 30-40, 40-54, 55+


def age_to_age_group(age):
    for (i, ag) in enumerate(AGE_GROUP):
        if age <= ag:
            return i
    return len(AGE_GROUP)  #55+


def get_target(x):
    if x < LOW_AGE or x > HIGH_AGE:
        return 0
    return 1

# def load_data():
#     fb_df = pd.read_csv(fb_filename, sep=' ', header=None, names=['raw_uid', 'gender', 'year'])
#     # fb_hash_id_df = pd.read_csv(fb_hash_id_filename, compression='gzip')
#     fb_hash_id_df = pd.read_csv(fb_hash_id_filename, sep='\t', header=None, names=['raw_uid', 'cityHash64', 'user_id'])
#     df = pd.merge(fb_hash_id_df, fb_df, left_on='raw_uid', right_on='raw_uid', how='left')
#     df.dropna(inplace=True)
#
#     try:
#         df['age'] = CURRENT_YEAR - df['year']
#     except:
#         df['year'] = pd.to_datetime(df['year'], format='%Y-%m-%d', errors='coerce')
#         df = df[df['year'].notnull()]  # remove all rows that have incorrect year
#         df['age'] = (BASE_DATE - df['year']).astype('<m8[Y]')
#     df['age_group'] = df['age'].apply(lambda x: get_target(x))
#     df = df[['user_id', 'age_group']]
#     return df


def load_data():
    df = pd.read_csv(fb_hash_id_filename, sep=' ', header=None, names=['user_id', 'gender', 'year'],
                     dtype={'user_id': str})
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
    LOW_AGE = int(sys.argv[1])
    HIGH_AGE = int(sys.argv[2])
    print("Low age \t {} ------------ High age \t {}".format(LOW_AGE, HIGH_AGE))
    fb_hash_id_filename = 'external_data/facebook_hash_id.csv'
    df = load_data()
    df.to_csv(f'external_data/new_age_label_{LOW_AGE}_{HIGH_AGE}.csv', index=False)
    print(df['age_group'].value_counts(normalize=True, sort=False))
    print(df.shape)
    print(df.head(20))
