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
sys.path.append("src/python/db")

from utils import make_directories
from mongodb import Mongodb
from joinData import join_and_save
from newLabelDefinition import process_new_target_with_conditions


HOST = "localhost"
PORT = 27017
DB_NAME = "adstarget"
DB_USERNAME = "adstarget-dev"
DB_PASSWORD = "M8bmU7CB9G3ItBxOOzck"
COLLECTION_NAME = "campaigns"


def get_status(config):
    status = dict(Name=config["name"],
                  Type="Model" if config["is_runnable"] else "Direct",
                  Status=1,
                  StartDate=datetime.strptime(config["start_date"], "%Y-%m-%d"),
                  EndDate=datetime.strptime(config["end_date"], "%Y-%m-%d"),
                  LastUpdated=None,
                  Active=None,
                  Config=config
                  )
    return status


def process(config, filepath):
    command = "{}/scripts/customTargeting.sh".format(WORKING_DIRECTORY)

    print('--------------Create new label----------')
    process_new_target_with_conditions(config['age_range'], NEW_LABEL_FILE)

    print("------------- Main program-------------")
    subprocess.call([command, CURRENT_DATE.strftime("%Y-%m-%d"), WORKING_DIRECTORY, FEATURE_DIRECTORY,
                     CAMPAIGN_DIRECTORY, config['cateID'], HYPER_PARAMS_DIRECTORY, NEW_LABEL_FILE,
                     config['metric'], config['cateID2']])

    file_result = os.path.join(CAMPAIGN_DIRECTORY, "prediction", "{}.gz".format(CURRENT_DATE.strftime("%Y-%m-%d")))
    if os.path.isfile(file_result):  # copy file to temporary folder to join
        shutil.copy(file_result, filepath)
    else:
        print('------ File : {} -------- NOT FOUND'.format(file_result))
        raise FileNotFoundError


if __name__ == '__main__':

    ap = argparse.ArgumentParser()
    ap.add_argument("-d", "--date", required=True, help="Date to run")
    ap.add_argument("-w", "--workspace", required=True, help="path to workspace")
    ap.add_argument("-f", "--feature", required=True, help="path to feature directory")
    ap.add_argument("-o", "--output", required=True, help="path to output directory")

    args = vars(ap.parse_args())

    CURRENT_DATE = datetime.strptime(args['date'], "%Y-%m-%d")
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

    active_campaigns = []
    status_campagins = []
    for jsonfile in list_jsons:
        config = json.load(open(jsonfile, 'r'))

        filepath = os.path.join(TEMPORARY_CUSTOM_TARGET_DIR, "{}.gz".format(config['name']))
        if not os.path.exists(filepath):
            config['name'] = os.path.splitext(config['name'])[0]
            filepath = os.path.join(TEMPORARY_CUSTOM_TARGET_DIR, "{}.gz".format(config['name']))

        date_campaign = datetime.strptime(config['end_date'], "%Y-%m-%d").date()
        status_camp = get_status(config)

        if CURRENT_DATE.date() <= date_campaign:
            status_camp["Active"] = 1
            status_camp["LastUpdated"] = CURRENT_DATE
            if config['is_runnable']:  # if this campaign needs train/predict procedure
                print("--------- Run campaign with json file : {}".format(jsonfile))
                CAMPAIGN_DIRECTORY = os.path.join(DATA_CAMPAIGNS_OUT_DIRECTORY, config['name'])
                make_directories(CAMPAIGN_DIRECTORY)
                HYPER_PARAMS_DIRECTORY = CAMPAIGN_DIRECTORY
                NEW_LABEL_FILE = os.path.join(CAMPAIGN_DIRECTORY, "label.gz")
                try:
                    process(config, filepath)
                except:  # Exception as err:
                    status_camp["Status"] = 0
                    # raise err

            active_campaigns.append(filepath)
        else:
            print("--------- Campaign: {} finished at {} -------------".format(config['name'], config['end_date']))
            status_camp["Active"] = 0
        status_campagins.append(status_camp)

    # join data
    # subprocess.call(["python", "src/python/main/joinData.py", f"-d{TEMPORARY_CUSTOM_TARGET_DIR}",
    #                  f"-o{os.path.join(FINAL_CUSTOM_TARGET_DIR, CUSTOM_TARGET_NAME)}"])
    print("Number of active campaigns --------  {}".format(len(active_campaigns)))
    join_and_save(active_campaigns, os.path.join(FINAL_CUSTOM_TARGET_DIR, CUSTOM_TARGET_NAME))

    # upload status to Mongodb
    mongodb = Mongodb(host=HOST, port=PORT, user_name=DB_USERNAME, password=DB_PASSWORD, dbname=DB_NAME)
    mongodb.insert_data(collection_name=COLLECTION_NAME, data=status_campagins)
