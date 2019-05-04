#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on May 03 17:34 2019

@author: phongdk
"""

import os
import json
import glob
import subprocess
from datetime import datetime, timedelta
import time
import shutil
from utils import make_directories

CURRENT_DATE = datetime.today().date() - timedelta(days=5)
WORKING_DIRECTORY = "/home/phongdk/workspace/customTargeting"
CAMPAIGNS_DIRECTORY = "/home/phongdk/data_custom_targeting/campaigns"
TEMPORARY_CUSTOM_TARGET_DIR = "/home/phongdk/tmp/cate_tmp"
FINAL_CUSTOM_TARGET_DIR = "/home/phongdk/tmp/categories-data"

# date=$1
# workspacePath=$2
# featurePath=$3
# outputPath=$4
# cateID=$5
# hyperParams=$6
# newLabels=$7
# metric=$8


def process(config):
    return
    command = "{}/scripts/customTargeting.sh".format(WORKING_DIRECTORY)
    subprocess.call([command, CURRENT_DATE.strftime("%Y-%m-%d"), WORKING_DIRECTORY, config['feature_path'],
                     config['output_path'], config['cateID'], config['hyperParams'], config['new_label_path'],
                     config['metric']])

    file_output = os.path.join(config['output_path'], "prediction", "{}.gz".format(CURRENT_DATE.strftime("%Y-%m-%d")))
    if os.path.isfile(file_output):  # copy file to temporary folder to join
        shutil.copy(file_output, os.path.join(TEMPORARY_CUSTOM_TARGET_DIR, config['name']))


if __name__ == '__main__':
    list_jsons = glob.glob(os.path.join(CAMPAIGNS_DIRECTORY, "*.json"))
    print("Process campaigns for date : {}".format(CURRENT_DATE))
    print(len(list_jsons))

    make_directories(TEMPORARY_CUSTOM_TARGET_DIR)
    make_directories(FINAL_CUSTOM_TARGET_DIR)

    for jsonfile in list_jsons:
        config = json.load(open(jsonfile, 'r'))
        date_campaign = datetime.strptime(config['end_date'], "%Y-%m-%d").date()
        if CURRENT_DATE < date_campaign:
            print("--------- Run campaign with json file : {}".format(jsonfile))
            if config['is_runnable']:  # if this campaign needs train/predict procedure
                process(config)
            else:
                print(jsonfile)
        else:
            print("-------- Campaign: {} finished at {} -------------".format(config['name'], config['end_date']))
            filepath = os.path.join(TEMPORARY_CUSTOM_TARGET_DIR, config['name'])
            if os.path.isfile(filepath):    # remove file if file is out-of-date
                os.remove(filepath)

    # join data
