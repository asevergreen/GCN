#!/usr/bin/env python 
# -*- coding:utf-8 -*-

import numpy as np
from Constants import *
from doc2vec import embeddingTextList
import random
import time

# 创建{event_i:label}字典
# 和{label:[event_i, event_j..]}字典
def create_event_label_dic():
    event_dic={}
    label_dic = {}
    for i in range(EVENT_NUM):
        f = open(TRAIN_DATA_DIR+"event_"+str(i)+".txt","r")
        event = eval(f.readlines()[0])
        event_dic["event_"+str(i)] = event["Label"]
        # if event["Label"] is None:
        #     continue
        try:
            label_dic[event["Label"]].append(i) # 只记录id
        except:
            label_dic[event["Label"]] = [i]
        f.close()
    del label_dic[None]
    del_rec = []
    for key in label_dic.keys():
        if "#" not in key or len(label_dic[key])<=1:
            del_rec.append(key)
    for key in del_rec:
        del label_dic[key]
    with open(INTERMEDIATE_DATA_DIR+"event.txt", "w") as f:
        f.write(str(event_dic))
    with open(INTERMEDIATE_DATA_DIR+"label.txt", "w") as f:
        f.write(str(label_dic))

# 根据label-event的对应关系和xdata创建1000个diff pair
def create_diff(num = 1000):
    with open(INTERMEDIATE_DATA_DIR +"label.txt","r") as f:
        label_dic = eval(f.read())

    labelNum = len(label_dic.keys())
    label_keys = list(label_dic.keys())
    diff = set()
    while len(diff) < num:
        ri, rj = random.sample(range(labelNum), 2)
        ei = random.choice(label_dic[label_keys[ri]])   # left-pair id
        ej = random.choice(label_dic[label_keys[rj]])   # right-pair id
        diff.add((ei,ej))
    # transform to array
    # xdata = np.load(INTERMEDIATE_DATA_DIR+"xdata.npy")
    # diff_data = []
    # for ei,ej in diff:
    #     diff_data.append([xdata[ei],xdata[ej]])
    diff_data = []
    for ei, ej in diff:
        diff_data.append([ei,ej])
    np.save(INTERMEDIATE_DATA_DIR+"diff.npy", np.array(diff_data))

# 根据label-event的对应关系和xdata创建1000个same pair
def create_same(num = 1000):
    with open(INTERMEDIATE_DATA_DIR+"label.txt", "r") as f:
        label_dic = eval(f.read())
    same = set()
    labelNum = len(label_dic.keys())
    label_keys = list(label_dic.keys())
    while len(same) < num:
        ri = random.randint(0,labelNum-1)
        leni = len(label_dic[label_keys[ri]])
        if leni < 2:
            continue
        ei,ej = random.sample(range(leni), 2) # index
        ei = label_dic[label_keys[ri]][ei]  # left-pair id
        ej = label_dic[label_keys[ri]][ej]
        same.add((ei,ej))
    # transform to array
    # xdata = np.load(INTERMEDIATE_DATA_DIR +"xdata.npy")
    # same_data = []
    # for ei,ej in same:
    #     same_data.append([xdata[ei],xdata[ej]])
    same_data = []
    for ei, ej in same:
        same_data.append([ei,ej])
    np.save(INTERMEDIATE_DATA_DIR+"same.npy", same_data)

# 提取出event中包括内容、转赞评、时间工具等信息放入text_list，将其embedding并保存程xdata.npy
def create_xdata():
    contents = []
    for i in range(EVENT_NUM):
        with open(TRAIN_DATA_DIR+"event_"+str(i)+".txt", "r") as f:
            event = eval(f.readlines()[0])
            content = event["Content"]+' '+str(event["Transfer"])+" "+str(event["Like"])+' '+str(event['Comment'])+" "+ event['PubTime']
            contents.append(content)
    embeddingTextList(contents)



# main
def format():
    print("begin formatting ...")
    start_time = time.time()
    create_event_label_dic()
    create_xdata()
    create_diff(DATA_SIZE)
    create_same(DATA_SIZE)
    print("Done!")
    delt = time.time()-start_time
    with open(TIME_REC, "a") as f:
        f.write("Formatting and Embedding COST:\t %.3f min(%.0f s)"%(delt/60, delt))

if __name__ == "__main__":
    format()