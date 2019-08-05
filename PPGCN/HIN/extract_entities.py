#!/usr/bin/env python 
# -*- coding:utf-8 -*-
# from stanfordcorenlp import StanfordCoreNLP
import time
import os
import jieba
import jieba.posseg
import jieba.analyse
from extract_keywords import *

# nlp = StanfordCoreNLP("D:\\Programs\\stanford_corenlp",lang='zh')
#从文本中抽取出实体
def find_raw_entity(text):
    raw_entities = []
    # tokens = nlp.pos_tag(text) # 中文分词
    tokens = jieba.posseg.cut(text) # 中文分词
    for token in tokens:
        try:
            if (token.flag=="nt") or (token.flag=="nz"): # 机构团体或其他专名
                raw_entities.append(token.word)
            elif token.flag=="nr" or token.flag=="ns": # 人名或地名
                raw_entities.append(token.word)    # 相当于pynlpir中的time word
            elif token.flag == "n": # 名词
                raw_entities.append(token.word)
        except:
            continue
    # ret list
    return list(set(raw_entities))

# 提取实体和关键词
# 对所有event文件应用find_raw_entity和extract_raw_keywords
def extract_raw_from_events():
    start_time = time.time()
    for i in range(EVENT_NUM):
        f = open(TRAIN_DATA_DIR+"event_"+str(i)+".txt", "r")
        event_content=eval(f.readlines()[0])    # 与数据库中相同格式的字典
        try:
            raw_entities=find_raw_entity(event_content["Content"])
        except:
            raw_entities = []
        try:
            raw_keywords = extract_raw_keywords(event_content["Content"])
        except Exception as e:
            raw_keywords = []
        #     print(e)
        f.close()
        with open(TRAIN_DATA_DIR+"event_"+str(i)+".txt", "a") as f:
            f.write("\n"+str(raw_entities).replace('\'', '').replace('[', '').replace(']', '').replace(',', ''))
            f.write("\n"+str(raw_keywords).replace('\'', '').replace("[", '').replace(']', '').replace(',', '')+"\n")   # str(list)自带空格

    print('Extract raw entities and keywords Done!!!')
    # 记录一下这个过程花费的时间
    with open(TIME_REC, 'a') as f:
        delt = (time.time() - start_time)
        f.write('Extracting all:\t %.3f min(%.0f s)\n' % ( delt/ 60, delt))

if __name__ == "__main__":
    extract_raw_keywords("大雨过后现彩虹，转发好运~")

