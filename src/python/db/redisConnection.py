#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mar 15 15:50 2019

@author: phongdk
"""

import redis


def connectRedis():
    r = redis.Redis(host='localhost', password='66d794ea7283270ab03f27017ee2936af89bda2980c95f6b124d4fa1e657d86b',
                    port=6379)
    return r


def get_browser_id(r, hash_id):
    result = r.hget('uid_mapping', hash_id)
    if result is not None:
        return result.decode("utf-8").split("|")[0]
    return None


if __name__ == '__main__':
    r = connectRedis()
    print(get_browser_id(r, "-6472122476610619770"))
