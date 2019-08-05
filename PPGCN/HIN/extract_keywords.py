#!/usr/bin/env python 
# -*- coding:utf-8 -*-

# from textrank4zh import TextRank4Keyword
from Constants import *
import synonyms
from synonyms.jieba import analyse



# 使用textrank4zh从文本中提取关键词。
# 每段文本提取10个,改动请在Constants.py设置
'''
def extract_raw_keywords(text, num = KEY_NUM_OF_EVERY_EVENT):
    tr4w = TextRank4Keyword()
    tr4w.analyze(text=text, lower=True, window=2)
    keywords=set()
    for item in tr4w.get_keywords(num, word_min_len=2):
        keywords.add(item.word)
    return list(keywords)
'''


def extract_raw_keywords(text, num = KEY_NUM_OF_EVERY_EVENT):
    # print("*"+text+"*")
    num = min(int(len(text)/5), num)
    tr = analyse.TextRank()
    tr.span = 3 # 默认span（滑动窗口大小）=5
    return tr.textrank(text, topK=num, allowPOS=('ns', 'n', 'vn', 'v','nz','nr'))

if __name__ =="__main__":
    print(extract_raw_keywords("大雨过后，抬头现彩虹[彩虹]转发好运～ "))
    print(extract_raw_keywords("大雨过后，抬头现彩虹[彩虹]转发好运～"))
    print(synonyms.compare("雨过后","彩虹出现"))
    # print(synonyms.seg("中文近义词工具包"))