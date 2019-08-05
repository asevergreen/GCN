#!/usr/bin/env python 
# -*- coding:utf-8 -*-

from Constants import *
import json
import time
# from extract_keywords import extract_raw_keywords

# mention2entity
def build_m2e_and_kb(entity_set):
    # m2e
    print('Start building m2e&kb dict ...')
    m2e = {}
    kb={}
    # build m2e
    i=0
    f = open(KG_DATA+'m2e.txt', 'r', encoding="utf-8")
    for line in f:  # 这种方式会读入行尾的'\n'
        i += 1
        m,e =line[:-1].split("\t")
        entity_set.add(e)
        try:
            m2e[m].append(e)
        except:
            m2e[m] = [e]
    f.close()
    print("m2e size:",i)
    # build kb, 无自环
    for m in m2e.keys():
        entities = m2e[m]
        # 建立近义词列表（mention相同）
        for e in entities:
            kb[e] = entities
            kb[e].remove(e)
    # save data
    with open(KG_DATA+"m2e.json","w") as json_file:
        json.dump(m2e, json_file)
    with open(KG_DATA+"kb.json","w") as json_file:
        json.dump(kb, json_file)
    print("Done!")


# description
def build_desc(entity_set):
    # desc
    print("Start building desc dict ...")
    desc = {}
    i=0
    f = open(KG_DATA+"baike_triples.txt", "r", encoding="utf-8")
    for line in f:
        e,attr,value=line[:-1].split("\t")
        if e not in entity_set:
            continue
        try:
            desc[e].append(value)
        except:
            desc[e] = [value]
        # desc[e] |= set(extract_raw_keywords(value,3))    # 求并集
        if i%1000000 == 0:
            print("read ",i)
        i+=1
    f.close()
    print("baike_triples size:",i)
    with open(KG_DATA+"desc.json","w") as json_file:
        json.dump(desc, json_file)

# build_main
def build_main():
    start_time = time.time()
    tmp_entity_set=set()
    build_m2e_and_kb(tmp_entity_set)
    build_desc(tmp_entity_set)
    # 记录消耗时间
    print('building kb|m2e|desc Done!!!')
    with open(TIME_REC, 'w') as f:
        delt = (time.time() - start_time)
        f.write('build kb|m2e|desc:\t %.3f min(%d s)\n' % (delt/60, delt))

if __name__ == "__main__":
    build_main()