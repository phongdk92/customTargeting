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


def load_data(filename):
    try:
        df = pd.read_csv(filename, sep=' ', header=None, names=['browser_id', 'category_id'], 
                         dtype={'browser_id': str, 'category_id': str})
        return df
    except:
        print('--------Load data fail ----- {}'.format(filename))
        return []


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument("-d", "--directory", required=True, help="path to customTargeting files")
    ap.add_argument("-o", "--output", required=True, help="output file")

    args = vars(ap.parse_args())

    directory = args['directory']
    output = args['output']
    dfs = [load_data(filename) for filename in sorted(glob.glob(directory + "*.gz"))]
    df = pd.concat(dfs, axis=0)
    # df['category_id'] = df['category_id'].astype(np.int16)
    print("Data shape : {}".format(df.shape))
    df.to_csv(output, sep=' ', compression='gzip', index=False, header=None)

