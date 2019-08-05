#!/usr/bin/env python 
# -*- coding:utf-8 -*-

import gensim
import jieba
import pandas as pd
import os
import time
import numpy as np
from gensim.models.doc2vec import Doc2Vec
from Constants import *

def getEventData():
    contents = []
    for i in range(1000):
        with open(TRAIN_DATA_DIR + "event_" + str(i) + ".txt", "r") as f:
            event = eval(f.readlines()[0])
            content = event["Content"] + ' ' + str(event["Transfer"]) + " " + str(event["Like"]) + ' ' + str(
                event['Comment']) + " " + event['PubTime']
            contents.append(content)
    return contents
def getText():
    # df_train = pd.read_excel(VEC_DIR + "10thousandsData.xlsx")[:3000]
    # content_train = list(df_train["Content"])
    content_train = getEventData()
    print("get Text!")
    return content_train

def cut_sentence(text):
    print("Start cutting sentence....")
    stop_list = l=[line[:-1] for line in open(VEC_DIR + "中文停用词.txt")]
    result = []
    errorNum = 0
    for each in text:
        try:
            each_cut = jieba.cut(each)
            each_split = ' '.join(each_cut).split()
            each_result = [word for word in each_split if word not in stop_list]
            result.append(' '.join(each_result))
        except Exception as e:
            print(text, ":", e)
            errorNum+=1
    print("Cut sentences done! (error:",errorNum)
    return result

def X_train(cut_sentence):
    print("Start make X_tain documents...")
    x_train=[]
    TaggededDocument = gensim.models.doc2vec.TaggedDocument
    for i,text in enumerate(cut_sentence):
        word_list = text.split(" ")
        l = len(word_list)
        word_list[l-1] = word_list[l-1].strip()
        document = TaggededDocument(word_list, tags=[i])
        x_train.append(document)
    print("Done!")
    print("totally train: ",len(x_train))
    return x_train

def train(x_train, size = 1000):
    print("training.....")
    model = Doc2Vec(x_train, min_count=1, window=3, vector_size=size, sample=1e-3, nagative=5,workers=4)
    model.train(x_train, total_examples=model.corpus_count, epochs=10)
    print("Training OK!")
    return model

def trainModel():
    model = train(X_train(cut_sentence(getText())))
    model.save(VEC_DIR + "doc2vecModel.model")
    print("SAVED!")

'''
def embeddingText(text,model):
    return model.infer_vector(cut_sentence(text).split(), alpha=0.025,steps=500)
'''

def embeddingTextList(text_list):
    model = Doc2Vec.load(VEC_DIR + "doc2vecModel.model")
    data_list = list()
    text_list = cut_sentence(text_list)
    for text in text_list:
        data_list.append(model.infer_vector(doc_words=text.split(" "),alpha=0.025, steps=500))
    data_list = np.array(data_list)
    np.save(INTERMEDIATE_DATA_DIR+"xdata.npy", data_list)

if __name__ == "__main__":
    start_time = time.time()
    trainModel()
    # print(embeddingTextList(["大雨过后现彩虹，转发好运~"]))
    delt = time.time() - start_time
    print("TIME COST: %.3f min(%.0f s)"%(delt/60, delt))