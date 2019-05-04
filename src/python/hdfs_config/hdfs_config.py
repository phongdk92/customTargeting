#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Apr 12 10:33 2019

@author: phongdk
"""

import hdfs3

storage_options = {'host': 'hdfs://ads-target1v.dev.itim.vn',
                   'port': 8020,
                   'dfs.domain.socket.path': '/home/phuongdv/hadoopdata/hadoop-hdfs/dn_socket',
                   'user': 'phongdk'}

hdfs = hdfs3.HDFileSystem('ads-target1v.dev.itim.vn', port=8020,
                          pars={'dfs.domain.socket.path': '/home/phuongdv/hadoopdata/hadoop-hdfs/dn_socket'})

HDFS_PREFIX = "hdfs:"
