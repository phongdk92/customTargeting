#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mar 12 16:56 2019

@author: phongdk
"""

import subprocess
import pandas as pd
import numpy as np
import os
import gc
from datetime import datetime, timedelta

CONNECT_TO_AGGREGATOR = "clickhouse-client --progress --user=stats_webui " \
                        "--password=`cat /home/phongdk/.clickhouse_pw` --host=aggregator3v.dev.itim.vn --query "

CONNECT_TO_BROWSER_STAT = "clickhouse-client --user=default --host=browser-stat1v.dev.itim.vn --query "


def get_data_from_server(connect_to_server, query, external=""):
    command = connect_to_server + "\"{}\" ".format(query) + external
    # print(command)
    # exit()
    output = subprocess.check_output(command, shell=True)
    output = output.decode('utf-8', errors='ignore').split('\n')
    output = [x.split('\t') for x in output]
    return output[:-1]


def collect_user_hash_id(filename, from_date, end_date, filter_uid):
    print("PROCESS : {} --- {}".format(filename, filter_uid))
    external = "--external --file {} --name='temp_uid' --structure='url String'".format(filter_uid)
    output = []
    query = "SELECT " \
            "distinct(user_id), "\
            "cityHash64(user_id) as u, "\
            "bitAnd(u, (bitShiftLeft(toInt64(1), toInt64(63)) - 1)) - bitAnd(u, (bitShiftLeft(toInt64(1), toInt64(63)))) as hash_id "\
            "FROM " \
            "browser.clickdata " \
            "WHERE " \
            "event_date BETWEEN '{}' AND '{}' AND " \
            "user_id in temp_uid".format(from_date, end_date)
    output.extend(get_data_from_server(CONNECT_TO_AGGREGATOR, query, external))
    print(len(output))
    columns = ['raw_uid', 'cityHash64', 'user_id']
    export_to_csv(filename, output, columns)


def export_to_csv(filename, output, columns):
    print('number of rows : {}'.format(len(output)))
    df = pd.DataFrame.from_records(output)
    df.columns = columns
    df.to_csv(os.path.join(PATH, filename), compression='gzip', index=False)


if __name__ == '__main__':
    from_date = '2019-02-20'
    end_date = '2019-03-11'
    PATH = 'external_data'
    filter_uid = os.path.join(os.getcwd(), PATH,  'facebook_id.csv')
    collect_user_hash_id('facebook_hash_id.csv.gz', from_date, end_date, filter_uid)

