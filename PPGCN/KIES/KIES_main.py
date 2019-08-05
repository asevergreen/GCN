#!/usr/bin/env python 
# -*- coding:utf-8 -*-
import os
import numpy as np
from Constants import *

if __name__ == "__main__":
    l = []
    n = 0
    for filename in os.listdir(SIM_DIR):
        f = np.load(SIM_DIR + filename)
        l.append(f)
    adj_data = np.array(l)

    # print(adj_data[4])
    print(adj_data.shape)
    np.save(INTERMEDIATE_DATA_DIR+"adj_data.npy", adj_data)