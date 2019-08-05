#!/usr/bin/env python 
# -*- coding:utf-8 -*-

'''
 设置一些模块内的全局常量
'''
# 每个event能提取的关键词个数
KEY_NUM_OF_EVERY_EVENT = 5

# Directory
PROJECT_PATH = "E:/tasks/PPGCN/Data/"
TRAIN_DATA_DIR = PROJECT_PATH + "trainData/"
INTERMEDIATE_DATA_DIR = PROJECT_PATH + "intermediateData/"
KG_DATA = PROJECT_PATH + "kgData/"
TIME_REC = PROJECT_PATH + "time.txt"

SIM_DIR = PROJECT_PATH + "Sim/"

VEC_DIR = PROJECT_PATH+'xlsTrain/'

MODEL_DIR = PROJECT_PATH+"Model/"

# meta-path
meta_paths = [
    # ['events', 'topics', 'keywords', 'topics', 'events'],
    ['events', 'entities', 'keywords', 'entities', 'events'],
    # ['events', 'entities', 'topics', 'entities', 'events']
]
EVENT_NUM = 2105

# train
DATA_SIZE = 1000 # 就是diff和same对数
NODE_SIZE = EVENT_NUM