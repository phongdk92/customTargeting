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
import argparse
import shutil
import sys

sys.path.append("src/python/utils")

from utils import make_directories
from joinData import join_and_save


# CURRENT_DATE = datetime.today().date() - timedelta(days=2)

def process(config):
    command = "{}/scripts/customTargeting.sh".format(WORKING_DIRECTORY)

    subprocess.call([command, CURRENT_DATE.strftime("%Y-%m-%d"), WORKING_DIRECTORY, FEATURE_DIRECTORY,
                     CAMPAIGN_DIRECTORY, config['cateID'], HYPER_PARAMS_DIRECTORY, NEW_LABEL_FILE,
                     config['metric'], config['cateID2']])

    file_output = os.path.join(CAMPAIGN_DIRECTORY, "prediction", "{}.gz".format(CURRENT_DATE.strftime("%Y-%m-%d")))
    if os.path.isfile(file_output):  # copy file to temporary folder to join
        shutil.copy(file_output, os.path.join(TEMPORARY_CUSTOM_TARGET_DIR, "{}.gz".format(config['name'])))
    else:
        print('------ File : {} -------- NOT FOUND'.format(file_output))


if __name__ == '__main__':

    ap = argparse.ArgumentParser()
    ap.add_argument("-d", "--date", required=True, help="Date to run")
    ap.add_argument("-w", "--workspace", required=True, help="path to workspace")
    ap.add_argument("-f", "--feature", required=True, help="path to feature directory")
    ap.add_argument("-o", "--output", required=True, help="path to output directory")

    args = vars(ap.parse_args())

    CURRENT_DATE = datetime.strptime(args['date'], "%Y-%m-%d").date()
    WORKING_DIRECTORY = args['workspace']  # "/home/phongdk/workspace/customTargeting"
    FEATURE_DIRECTORY = args['feature']  # '/home/tuannm/target/demography/data'
    OUTPUT_DIRECTORY = args['output']  # "/home/phongdk/data_custom_targeting/"

    JSON_CAMPAIGNS_DIRECTORY = os.path.join(OUTPUT_DIRECTORY, "campaigns")
    DATA_CAMPAIGNS_OUT_DIRECTORY = os.path.join(OUTPUT_DIRECTORY, "data")
    TEMPORARY_CUSTOM_TARGET_DIR = os.path.join(OUTPUT_DIRECTORY, "cate_tmp")
    FINAL_CUSTOM_TARGET_DIR = os.path.join(OUTPUT_DIRECTORY, "categories-data")
    CUSTOM_TARGET_NAME = "adstarget_custom.gz"

    list_jsons = glob.glob(os.path.join(JSON_CAMPAIGNS_DIRECTORY, "*.json"))
    print("Process campaigns for date : {}".format(CURRENT_DATE))
    print(len(list_jsons))

    make_directories(TEMPORARY_CUSTOM_TARGET_DIR)
    make_directories(FINAL_CUSTOM_TARGET_DIR)

    for jsonfile in list_jsons:
        config = json.load(open(jsonfile, 'r'))
        date_campaign = datetime.strptime(config['end_date'], "%Y-%m-%d").date()
        if CURRENT_DATE < date_campaign:
            if config['is_runnable']:  # if this campaign needs train/predict procedure
                print("--------- Run campaign with json file : {}".format(jsonfile))
                CAMPAIGN_DIRECTORY = os.path.join(DATA_CAMPAIGNS_OUT_DIRECTORY, config['name'])
                make_directories(CAMPAIGN_DIRECTORY)
                HYPER_PARAMS_DIRECTORY = CAMPAIGN_DIRECTORY
                NEW_LABEL_FILE = os.path.join(CAMPAIGN_DIRECTORY, "label.csv")
                try:
                    process(config)
                except Exception as err:
                    raise err
        else:
            print("--------- Campaign: {} finished at {} -------------".format(config['name'], config['end_date']))
            filepath = os.path.join(TEMPORARY_CUSTOM_TARGET_DIR, "{}.gz".format(config['name']))
            if os.path.isfile(filepath):  # remove file if file is out-of-date
                os.remove(filepath)
                print(" ***** Remove file '{}' from temporary folder '{}'".format(config['name'], TEMPORARY_CUSTOM_TARGET_DIR))

    # join data
    # subprocess.call(["python", "src/python/main/joinData.py", f"-d{TEMPORARY_CUSTOM_TARGET_DIR}",
    #                  f"-o{os.path.join(FINAL_CUSTOM_TARGET_DIR, CUSTOM_TARGET_NAME)}"])
    join_and_save(TEMPORARY_CUSTOM_TARGET_DIR, os.path.join(FINAL_CUSTOM_TARGET_DIR, CUSTOM_TARGET_NAME))
