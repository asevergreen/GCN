#!/usr/bin/env python 
# -*- coding:utf-8 -*-

import json
import time
from Constants import *

# 添加实体和关键词的联系
#       两层for循环，如果key_i在entity_j的描述desc中，就利用entities和keywords标记两者的关系
#       最后set去重
def add_key_entity(entities, keywords, full_entities, full_keywords, desc):
    num_keywords = len(full_keywords)
    num_entities = len(full_entities)
    print('Totally ', num_keywords)
    for i in range(num_keywords):
        key = full_keywords[i]
        for j in range(num_entities):
            e = full_entities[j]
            try:
                if key in desc[e]:
                    keywords[key].append(e)
                    entities[e].append(key)
            except:
                continue
    for k in full_keywords:
        keywords[k] = list(set(keywords[k]))
    for e in full_entities:
        entities[e] = list(set(entities[e]))
    with open(INTERMEDIATE_DATA_DIR + 'keywords.json', 'w') as json_file:
        json.dump(keywords, json_file)
    with open(INTERMEDIATE_DATA_DIR + 'entities.json', 'w') as json_file:
        json.dump(entities, json_file)


# 为创建key-entity关系load数据，并调用add_key_entity
def add_KE_relation():
    start_time = time.time()
    with open(INTERMEDIATE_DATA_DIR+"keywords.json","r") as json_file:
        keywords = json.load(json_file)
    with open(INTERMEDIATE_DATA_DIR + "entities.json","r") as json_file:
        entities = json.load(json_file)
    # desc = {}
    with open(KG_DATA + "desc.json", "r") as json_file:
        desc = json.load(json_file)
    # add relationships
    full_entities = list(entities.keys())
    full_keywords = list(keywords.keys())
    print('Adding key entity relations ...')
    add_key_entity(entities, keywords, full_entities, full_keywords, desc)

    print('Done!!!')
    with open(TIME_REC, 'a') as f:
        delt = (time.time() - start_time)
        f.write('Adding key-entity relations:\t %.3f min(%.0f s)\n' % (delt/ 60, delt))