#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on May 16 10:43 2019

@author: phongdk
"""

from pymongo import MongoClient
from datetime import datetime


class Mongodb(object):
    def __init__(self, host, port, user_name, password,  dbname="testdb"):
        self.host = host
        self.port = port
        self.dbname = dbname
        self.client = MongoClient("mongodb://{}:{}@{}:{}/{}".format(user_name, password, host, port, dbname))

    def insert_data(self, collection_name, data):
        print("---------------------Insert data to {}.{}------------------------".format(self.dbname, collection_name))
        mycollection = self.client[self.dbname][collection_name]
        mycollection.insert_many(data) if isinstance(data, list) else mycollection.insert_one(data)
        # for x in mycollection.find():
        #     print(x)

    def drop_collection(self, collection_name):
        print("-----------------Drop collection {} from db {}-------------------".format(collection_name, self.dbname))
        self.client[self.dbname][collection_name].drop()

    def list_data(self, collection_name):
        mycollection = self.client[self.dbname][collection_name]
        for x in mycollection.find():
            print(x)


if __name__ == '__main__':
    HOST = "localhost"
    PORT = 27017
    DB_NAME = "adstarget"
    DB_USERNAME = "adstarget-dev"
    DB_PASSWORD = "M8bmU7CB9G3ItBxOOzck"
    COLLECTION_NAME = "campaigns"

    mongodb = Mongodb(host=HOST, port=PORT, user_name=DB_USERNAME, password=DB_PASSWORD, dbname=DB_NAME)

    my_dict = dict(Name="test",
                   Type="Model",
                   Status="1",
                   StartDate=datetime.strptime("2019-04-05", "%Y-%m-%d"),
                   EndDate='2019-05-08',
                   LastUpdated='2019-05-06',
                   Active="1"
                   )
    print(my_dict)
    # mongodb.insert_data(collection_name=COLLECTION_NAME, data=my_dict)
    mongodb.drop_collection(collection_name=COLLECTION_NAME)
    # mongodb.list_data(collection_name="campaigns")
    # print(mongodb.client[DB_NAME].list_collection_names())