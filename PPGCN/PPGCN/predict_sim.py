#!/usr/bin/env python 
# -*- coding:utf-8 -*-

from PPGCN import *

def predit_similarity():
    with tf.Session() as sess:
        tf.train.Saver.restore(sess, MODEL_DIR+"model")
