#encoding:utf-8  实现文本分类之朴素贝叶斯
import sys
import os
import re
import time
import math
import random
from sklearn import preprocessing
from sklearn.cross_validation import train_test_split
import numpy as np
import scipy as sp
from classes import readfile
def Naive_Bayes(Dic_train,test_sen): #基于朴素贝叶斯的多项式模型，计算test_sen的类别
    labels = ['acq','corn','crude','earn','grain','interest','money-fx','ship','trade','wheat']
    label_result = []
    Pro_value = []
    total_wordnum = 0
    test_sen = test_sen.split()
    for label in labels:
        for sen in Dic_train[label]:
            sen = sen.split()
            total_wordnum = total_wordnum + len(sen)
    for label in labels:
        num_label = 0
        Pro_value = []
        for sen in Dic_train[label]:
            sen_list = sen.split()
            num_label = num_label + len(sen_list)  #一个类别下的单词总数
        for sen_test in test_sen:
            num_word = 0
            for sen in Dic_train[label]:
                sen_list = sen.split()
                num_word = num_word + sen_list.count(sen_test) #该类别下包含某单词的个数
            pro = math.log((num_word+1)*1.0/num_label)
            pro = round(pro,4)
            Pro_value.append(pro)
        pro_label = math.log((num_label*1.0)/total_wordnum)
        pro_label = round(pro_label,4)
        Pro_value.append(pro_label)
        final_result = round(sum(Pro_value),4)
        label_result.append(final_result)
    return labels[label_result.index(max(label_result))]
def Bayes_result(Dic_train,Dic_test):
    labels = ['acq','corn','crude','earn','grain','interest','money-fx','ship','trade','wheat']
    start = time.clock()
    name_file = {}
    right_num,allsen_num = 0,0
    path_now = os.getcwd()
    path_test = path_now + '\\'+'test'
    for file_name in os.listdir(path_test):  #同一个文件可能包含在不同的类中，判断出一个就判定为正确
        class_name = file_name
        text_path = path_test + '\\' + file_name
        for text_name in os.listdir(text_path):
            name_file.setdefault(class_name,[]).append(text_name)
    for label in labels:
        count,correct_num = 0,0
        for sen in Dic_test[label]:
            sen_name = name_file[label][count]  #取当前文件的文件名
            B_label = Naive_Bayes(Dic_train,sen)
            if (B_label == label) or (sen_name in name_file[B_label]):
                correct_num = correct_num + 1
            count = count + 1
            allsen_num = allsen_num + 1
        right_num = right_num + correct_num
        print '在类别',label,'中的正确率：',correct_num*1.0/len(Dic_test[label])
    print '整体的正确率：',right_num*1.0/allsen_num
    end = time.clock()
    print "run: %f s" % (end-start)
    return
if __name__=='__main__':
    Dic_train = {}
    Dic_test = {}
    Dic_train = readfile('training')
    Dic_test = readfile('test')
    Bayes_result(Dic_train,Dic_test)