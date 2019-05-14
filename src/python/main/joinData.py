#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mar 25 17:17 2019

@author: phongdk
"""

import numpy as np
import pandas as pd
import argparse
import glob
import os


def load_data(filename):
    print(filename)
    try:
        df = pd.read_csv(filename, sep=' ', header=None, names=['browser_id', 'category_id'],
                         dtype={'browser_id': str, 'category_id': str})
        print(df.shape)
        return df
    except:
        print('--------Load data fail ----- {}'.format(filename))
        return []


def join_and_save(filenames, output):
    dfs = [load_data(filename) for filename in filenames]
    try:
        df = pd.concat(dfs, axis=0)
        # df['category_id'] = df['category_id'].astype(np.int16)
        print("Data shape : {}".format(df.shape))
        assert df.shape[1] == 2
        df.to_csv(output, sep=' ', compression='gzip', index=False, header=None)
    except ValueError as e:
        print('ERROR : ------------ Cannot concate data frames')
        raise e


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument("-d", "--directory", required=True, help="path to customTargeting files")
    ap.add_argument("-o", "--output", required=True, help="output file")

    args = vars(ap.parse_args())

    directory = args['directory']
    output = args['output']

    filenames = sorted(glob.glob(os.path.join(directory, "*.gz")))
    join_and_save(filenames, output)
