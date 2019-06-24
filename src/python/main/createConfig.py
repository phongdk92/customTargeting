# !/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on May 03 17:00 2019

@author: phongdk
"""

import json
import os

CAMPAIGNS_DIR = "/home/phongdk/data_custom_targeting/campaigns"

if __name__ == '__main__':
    config = dict(name='test#1234',
                  start_date='2019-04-10',
                  end_date='2019-04-27',
                  is_runnable=True,
                  cateID='000000',
                  metric='precision',
                  cateID2='0',
                  age_range=["30-44"]
                  )

    # config = dict(name='samsung#595930',
    #               start_date='2019-05-08',
    #               end_date='2019-05-10',
    #               is_runnable=False,
    #               )

    outJson = os.path.join(CAMPAIGNS_DIR, "{}.json".format(config['name']))
    with open(outJson, 'w') as fp:
        json.dump(config, fp)
