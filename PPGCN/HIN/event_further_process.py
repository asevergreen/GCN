#!/usr/bin/env python 
# -*- coding:utf-8 -*-

import json
import synonyms
from extract_entities import *


# 替换掉语义不明的分词，转换成最可能的entity
def replace_ambiguous(context, possible_entities):
    if(len(possible_entities)==0):
        return ''
    if len(possible_entities) == 1:
        return possible_entities[0]
    score = []
    #    print(possible_entities)
    for e in possible_entities:
        # ignore: 是否忽略OOV(out of vocabulary,未登录词，即训练时没有测试时遇到的词)，False时，随机生成一个向量
        score.append(synonyms.compare(context, e, ignore=True))
    # 返回最相似的entity //score.index()返回下标
    return possible_entities[score.index(max(score))]

# 将分词彻底转换成实体
def extract_entities(m2e, r_e, eventNum):
    tmp_entities = {}   # 一个实体对应的所有event
    tmp_events = {}     # 一个event对应的所有实体
    # init
    for i in range(eventNum):
        tmp_events['event_'+str(i)] = []
    # 构建entity，keywords之间的关系。
    # 用r_e,m2e两个字典根据raw_entities获取可能的实体添加进空列表
    key_set = set()
    for i in range(eventNum):
        f = open(TRAIN_DATA_DIR+'event_'+str(i)+'.txt', 'r')
        content = f.readlines()
        keywords = content[2].strip().replace(' ', ',') # 第2行放的是用' '分隔的keywords（从0数）
        key_set |= set(keywords.split(","))
        context = eval(content[0])["Content"].strip()    # 微博内容
        try:
            raw_entities = content[1].strip().split(' ')    # 第1行是分词结果，前步提取的raw实体 //可能没有，所以用try括起来
        except:
            continue
#        print(context)
        for r in raw_entities:
            try:
                e = r_e[r]  # r是一个str，说明r_e是一个字典，这个是分词对应的实体
            except:
                try:
                    # 如果不能在r_e中得到r对应的entity，尝试来m2e中获取同义词列表
                    possible_entities = m2e[r]
                except:
                    continue
                e = replace_ambiguous(context+keywords, possible_entities)
                if e=='':
                    continue
                r_e[r] = e
            tmp_events['event_'+str(i)].append(e)    # eventxxx文件中可能包含的实体
            try:
                tmp_entities[e].append('event_'+str(i))
            except:
                tmp_entities[e] = [ 'event_'+str(i)]

        del content, context, raw_entities, keywords
        f.close()
    # 转换为字典
    key_dic={}
    for key in key_set:
        key_dic[key] = []
    # 记录所有出现过的keywords
    with open(INTERMEDIATE_DATA_DIR + "keywords.json", "w") as json_file:
        json.dump(key_dic, json_file)
    # 记录entity - events关系列表
    with open(INTERMEDIATE_DATA_DIR+'entities.json', 'w') as json_file:
        json.dump(tmp_entities, json_file)
    # 记录event - entities关系列表
    with open(INTERMEDIATE_DATA_DIR+'events.json', 'w') as json_file:
        json.dump(tmp_events, json_file)
    # 经过补充后的[分词-实体]替换的词典，在抽取entity时用到
    with open(KG_DATA+'r_e.json', 'w') as json_file:
        json.dump(r_e, json_file)

# main
def extract_main():
    start_time = time.time()
    # extract raw
    extract_raw_from_events()
    # load m2e
    with open(KG_DATA+'m2e.json', 'r') as json_file:
        m2e = json.load(json_file)
    # load r_e：[分词-实体]字典结构
    try:
        # 如果是第一次运行就没有r_e.json文件
        with open(KG_DATA + 'r_e.json', 'r') as json_file:
            r_e = json.load(json_file)
    except:
        r_e = {}
    print("extract entities...")
    # extract entities
    # extract_entities(m2e, r_e, len(os.listdir(TRAIN_DATA_DIR)))
    extract_entities(m2e, r_e, EVENT_NUM)

    # 时间记录
    print('Done!!!')
    with open(TIME_REC, 'a') as f:
        delt = (time.time() - start_time)
        f.write('extract entities|keywords: %.3f minutes(%.0f s)\n' % ( delt/ 60, delt))
