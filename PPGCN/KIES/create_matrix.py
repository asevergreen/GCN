#!/usr/bin/env python 
# -*- coding:utf-8 -*-

import numpy as np
import json
import time
from Constants import *

# create single matrix
def M_single_step(s, d):
    src = list(s.keys())
    dst = list(d.keys())
    M_l = np.zeros((len(src), len(dst)))
    for i in range(len(src)):
        for j in range(len(dst)):
            if dst[j] in s[src[i]]:
                M_l[i][j] = 1
    return M_l

# create matrix according to meta-path
# 根据meta-path建立相似度矩阵
def cal_M(meta_path):
    for i in range(len(meta_path) - 1):
        with open(INTERMEDIATE_DATA_DIR+meta_path[i]+'.json', 'r') as json_file:
            src = json.load(json_file)
        with open(INTERMEDIATE_DATA_DIR+meta_path[i+1]+'.json', 'r') as json_file:
            dst = json.load(json_file)
        M_l = M_single_step(src, dst)
        print('Single step matrix generated.')
        if i == 0:
            M = M_l
        else:
            M = np.matmul(M, M_l)   # 矩阵相乘
        del M_l
        print('Single step multiply process finished.')
        del src, dst
    return M

# calculate sim between two nodes
# 计算两个event的similarity 相似度
def know_sim(M, i, j):
    broadness = M[i][i] + M[j][j]
    overlap = 2*M[i][j]
    if broadness == 0:
        return 0
    else:
        return overlap/broadness


def create_sim_matrix():
    print('Start creating matrix ...')
    start_time = time.time()
    for k in range(len(meta_paths)):
        path = meta_paths[k]
        M = cal_M(path)
        sim = np.zeros((EVENT_NUM, EVENT_NUM))
        print('path'+str(k)+' M calculation finished.')
        for i in range(EVENT_NUM):
            sim[i][i] = 1
        for i in range(EVENT_NUM - 1):

            for j in range(i+1, EVENT_NUM):
                sim[i][j] = know_sim(M, i, j)
                sim[j][i] = sim[i][j]
        del M
        np.save(SIM_DIR+'sim_'+str(path)+'.npy', sim)
        del sim
        print('path'+str(k)+' sim matrix finished.')
    # 记录时间
    with open(TIME_REC, "a") as f:
        delt = time.time()-start_time
        f.write("create SIM matrix:\t %.3f min(%.0f s)\n"%(delt/60, delt))

if __name__ == "__main__":
    create_sim_matrix()