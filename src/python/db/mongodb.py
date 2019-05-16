#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on May 16 10:43 2019

@author: phongdk
"""

from pymongo import MongoClient


class Mongodb(object):
    def __init__(self, host="10.0.11.55", port=27017, dbname="testdb"):
        self.host = host
        self.port = port
        self.dbname = dbname
        self.client = MongoClient("mongodb://{}:{}/".format(host, port))

    def insert_data(self, collection_name, data):
        print("---------------------Insert data to {}.{}------------------------".format(self.dbname, collection_name))
        mycollection = self.client[self.dbname][collection_name]
        mycollection.insert_many(data) if isinstance(data, list) else mycollection.insert_one(data)
        # for x in mycollection.find():
        #     print(x)

    def drop_collection(self, collection_name):
        print("-----------------Drop collection {} from db {}-------------------".format(collection_name, self.dbname))
        self.client[self.dbname][collection_name].drop()


if __name__ == '__main__':
    mongodb = Mongodb(dbname="testdb")

    my_dict = dict(Name="topica",
                   Type="Model",
                   Status="1",
                   StartDate="2019-04-05",
                   EndDate='2019-05-08',
                   LastUpdated='2019-05-06',
                   Active="1"
                   )
    mongodb.insert_data(collection_name="information", data=my_dict)
    mongodb.drop_collection(collection_name="information")