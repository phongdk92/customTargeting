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
    # config = dict(name='P30_44#586936.gz',
    #               start_date='2019-03-26',
    #               end_date='2019-06-04',
    #               is_runnable=True,
    #               feature_path='/home/tuannm/target/demography/data',
    #               cateID='15905',
    #               metric='f1_score',
    #               output_path='/home/phongdk/tmp',
    #               hyperParams='best_params/30-44_20190604_50',
    #               new_label_path='external_data/new_age_label_30_44.csv',
    #               )

    config = dict(name='shopee#591498.gz',
                  start_date='2019-04-11',
                  end_date='2019-05-25',
                  is_runnable=False,
                  )

    outJson = os.path.join(CAMPAIGNS_DIR, config['name'].replace(".gz", ".json"))
    with open(outJson, 'w') as fp:
        json.dump(config, fp)
