#encoding:utf-8  实现文本分类之knn（k近邻）
import sys
import os
import re
import time
import math
import random
from math import sqrt
from sklearn import preprocessing
from sklearn import svm
from sklearn.cross_validation import train_test_split
import numpy as np
import scipy as sp
from classes import readfile
from classes import read_table
def Knn(Dic_train,Dic_test):
    file_tf=open('feature_tf.txt','r')
    file_tfidf=open('feature_tfidf.txt','r')
    vector_tf = []    #用tf表示向量
    vector_tfidf = []    #用tfidf表示向量
    test_vector = []    #测试集的文档向量化
    class_label = []  #训练文档的标签
    test_label = []  #测试文档的标签
    total_tf = file_tf.readlines()
    total_tfidf = file_tfidf.readlines()
    #生成训练文档类别标签
    path_now = os.getcwd()
    path_train = path_now + '\\'+'training'
    for file_name in os.listdir(path_train):
        text_path = path_train + '\\' + file_name
        for text_name in os.listdir(text_path):
            class_label.append(file_name)
    #生成测试文档类别标签
    path_test = path_now + '\\'+'test'
    for file_name in os.listdir(path_test):
        text_path = path_test + '\\' + file_name
        for text_name in os.listdir(text_path):
            test_label.append(file_name)

    #生成基于词频逆文档的文档向量
    for word_tfidf in total_tfidf:
        word_tfidf = word_tfidf.strip('\n')
        tfidf_list = word_tfidf.split(', ')
        tfidf_list= map(float, tfidf_list)
        vector_tfidf.append(tfidf_list)
    for index in range(len(class_label)):
        vector_tfidf[index].append(class_label[index])
    test_vector = Doc_vector(Dic_test)
    for index in range(len(test_label)):
        test_vector[index].append(test_label[index])
    Knn_result(vector_tfidf,test_vector,400)
    return
def Knn_result(vector_tfidf,test_vector,k_num): #取前k个值作为最近邻
    start = time.clock()
    final_num = {}
    name_file = {}
    len_train = len(vector_tfidf)
    len_test = len(test_vector)
    vector_len = len(test_vector[0])
    path_now = os.getcwd()
    path_test = path_now + '\\'+'test'
    for file_name in os.listdir(path_test):  #同一个文件可能包含在不同的类中，判断出一个就判定为正确
        class_name = file_name
        text_path = path_test + '\\' + file_name
        for text_name in os.listdir(text_path):
            name_file.setdefault(class_name,[]).append(text_name)
    print len(name_file['acq'])
    count,correct_num = 0,0
    label_old = 'acq'
    for index_test in range(len_test):
        knn_num = list()
        label = test_vector[index_test][vector_len-1]
        if label != label_old:
           count = 0
        doc_name = name_file[label][count]  #取当前文件的文件名
        for index_train in range(len_train):
            p = sim_distance_cos(test_vector[index_test][:vector_len-1],vector_tfidf[index_train][:vector_len-1])
            knn_num.append((p,vector_tfidf[index_train][vector_len-1]))
        knn_num.sort(reverse = True) #排序
        top_k = knn_num[0:k_num]       #取前k个
        di = dict()
        for l in top_k:
            s,cls = l
            times = di.get(cls)
            if not times:
                times = 0
            di[cls] = times + 1 #统计属于类别的文档数
        sortli = sorted(di.iteritems(),None,lambda d:d[1],True) #排序取文档数最多的类
        final_cls = sortli[0][0] #分出来的类别
        print '当前的类为：',label,"文档号为:",doc_name
        print '预测的类为：',final_cls
        if (final_cls == label) or (doc_name in name_file[final_cls]):
            correct_num = correct_num + 1
        count = count + 1
        print '预测正确的个数为：',correct_num
        label_old = label
    print '整体的正确率：',correct_num*1.0/len(test_vector)
    end = time.clock()
    print "KNN文本分类用时: %f s" % (end-start)
    #return cls == tocls
    return
def sim_distance_cos(p1,p2):    #余弦相似度
    ss = sum( [p1[index]*p2[index] for index in range(len(p1)) ] )
    sq1 = sqrt( sum( [pow(p1[index],2) for index in range(len(p1)) ] ) )
    sq2 = sqrt( sum( [pow(p2[index],2) for index in range(len(p1)) ] ) )
    if sq1 == 0 or sq2 == 0:
        p = 0.0
    else:
        p = round(float(ss)/(sq1*sq2),5)
    return p

def Doc_vector(Dic_test):
    start = time.clock()
    read_num = 0   #取每个类别中的前200维作为特征向量。
    class_num = ['0','1','2','3','4','5','6','7','8','9']
    labels = ['acq','corn','crude','earn','grain','interest','money-fx','ship','trade','wheat']
    vector_svm = []
    feature_tfidf = []  #词频逆文档量化
    list_alldoc = [] #存储所有的文档
    list_temp = []
    list_feature1 = []
    table_fre = {}
    feature_path = open("feature.txt",'r')
    file_chi = feature_path.readlines()
    for each_word in file_chi:
        if each_word.strip('\n') in class_num :
            if len(list_temp) != 0 :
                list_feature1.extend(list_temp[:200])
            list_temp =[]
        else:
            list_temp.append(each_word.strip('\n'))
    list_feature1.extend(list_temp[:200])
    list_feature = list(set(list_feature1))  #特征元素去重,不改变原来顺序
    list_feature.sort(key=list_feature1.index)
    #使用os.listdir获取路径
    path_now = os.getcwd()
    path_train = path_now + '\\'+'training'
    for file_name in os.listdir(path_train):
        text_path = path_train + '\\' + file_name
        for text_name in os.listdir(text_path):
            list_alldoc.append(text_name)
    table_fre = read_table()
    for label in labels:
        for test_sen in Dic_test[label]:
            temp_tf = []
            temp_tfidf = []
            train_sen = test_sen.split()
            for fea_word in list_feature:
                count_tf = test_sen.count(fea_word)
                numerator = len(list_alldoc)   #计算tfidf 的分子
                Denominator = 1 + sum(table_fre[fea_word][0])  #计算tfidf 的分母
                count_tfidf = round(count_tf*round(math.log(numerator*1.0/Denominator*1.0),4),4)
                temp_tfidf.append(count_tfidf)
            feature_tfidf.append(temp_tfidf)
    end = time.clock()
    print "生成测试集文档tfidf向量用时: %f s" % (end-start)
    return feature_tfidf
if __name__=='__main__':
    Dic_train = {}
    Dic_test = {}
    Dic_train = readfile('training')
    Dic_test = readfile('test')
    #Doc_vector(Dic_test)
    Knn(Dic_train,Dic_test)