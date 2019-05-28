#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mar 12 17:46 2019

@author: phongdk
"""

import numpy as np
import pandas as pd
import sys
from datetime import datetime
import argparse

CURRENT_YEAR = datetime.now().year
BASE_DATE = pd.Timestamp(datetime.now())
AGE_GROUP = [17, 29, 40, 54]  # 0-17, 18-29, 30-40, 40-54, 55+

fb_hash_id_filename = 'external_data/facebook_hash_id.csv'

def age_to_age_group(age):
    for (i, ag) in enumerate(AGE_GROUP):
        if age <= ag:
            return i
    return len(AGE_GROUP)  #55+


def get_target(x, low_age, high_age):
    if x < low_age or x > high_age:
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
    return df


def process_new_target(low_age, high_age, output):
    print("Low age \t {} ------------ High age \t {}".format(low_age, high_age))
    df = load_data()
    df['age_group'] = df['age'].apply(lambda x: get_target(x, low_age, high_age))
    df[['user_id', 'age_group']].to_csv(output, index=False)
    print(df['age_group'].value_counts(normalize=True, sort=False))


def get_target_with_conditions(x, conditions):
    for c in conditions:
        low_age, high_age = c.split(":")[-1].split("-")
        if int(low_age) <= x <= int(high_age):
            return 1
    return 0


def process_new_target_with_conditions(conditions, output):
    print("Conditions {}".format(conditions))
    df = load_data()
    df['age_group'] = df['age'].apply(lambda x: get_target_with_conditions(x, conditions))
    df[['user_id', 'age_group']].to_csv(output, index=False)
    print(df['age_group'].value_counts(normalize=True, sort=False))


if __name__ == '__main__':
    # ap = argparse.ArgumentParser()
    # ap.add_argument("-low", "--low", required=False, help="Low age", type=int, default=0)
    # ap.add_argument("-high", "--high", required=False, help="High age", type=int, default=np.inf)
    # ap.add_argument("-o", "--output", required=True, help="Output file", default="label.gz")
    # # ap.add_argument("-r", "--reverse", required=False, help="Reverse or not", type=int, default=0)
    # # if forcusing on 22- (X-) we have to inverse label (set Reverse=1), <22: 1, 22+: 0 instead of <22:0, 22+: 1
    #
    # # if focusing on 22-, set low=0, high=22
    # # if focusing on 22+, set low=22, high=np.inf
    # # This label is related to optimization F1-score on label 1
    #
    # args = vars(ap.parse_args())
    # LOW_AGE = args['low']   # int(sys.argv[1])
    # HIGH_AGE = args['high']
    # output = args['output']

    # process_new_target(low_age=LOW_AGE, high_age=HIGH_AGE, output=output)
    conditions = ["18-25", "30-45"]
    process_new_target_with_conditions(conditions, 'test.csv')
