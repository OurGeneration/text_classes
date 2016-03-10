#encoding:utf-8  实现文本分类之SVM
import sys
import os
import re
import time
import math
import random
from sklearn import preprocessing
from sklearn import svm
from sklearn.cross_validation import train_test_split
import numpy as np
import scipy as sp
from classes import readfile
def Svm_vector(Dic_train,Dic_test): #利用 tf 与 tfidf 量化特征向量，比较对比结果。
    start = time.clock()
    read_num = 0   #取每个类别中的前200维作为特征向量。
    class_num = ['0','1','2','3','4','5','6','7','8','9']
    labels = ['acq','corn','crude','earn','grain','interest','money-fx','ship','trade','wheat']
    file_tf=open('feature_tf.txt','w')
    file_tfidf=open('feature_tfidf.txt','w')
    vector_svm = []
    feature_tf = []  #词频量化
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
    print len(list_feature1)
    list_feature = list(set(list_feature1))  #特征元素去重,不改变原来顺序
    list_feature.sort(key=list_feature1.index)
    print len(list_feature)
    #使用os.listdir获取路径
    path_now = os.getcwd()
    path_train = path_now + '\\'+'training'
    for file_name in os.listdir(path_train):
        text_path = path_train + '\\' + file_name
        for text_name in os.listdir(text_path):
            list_alldoc.append(text_name)
    print len(list_alldoc)
    table_fre = read_table()
    print table_fre
    for label in labels:
        for train_sen in Dic_train[label]:
            temp_tf = []
            temp_tfidf = []
            train_sen = train_sen.split()
            for fea_word in list_feature:
                count_tf = train_sen.count(fea_word)
                numerator = len(list_alldoc)   #计算tfidf 的分子
                Denominator = 1 + sum(table_fre[fea_word][0])  #计算tfidf 的分母
                count_tfidf = round(count_tf*round(math.log(numerator*1.0/Denominator*1.0),4),4)
                temp_tf.append(count_tf)
                temp_tfidf.append(count_tfidf)
            feature_tf.append(temp_tf)
            feature_tfidf.append(temp_tfidf)
            file_tf.write(str(temp_tf).lstrip('[').rstrip(']'))
            file_tf.write('\n')

            file_tfidf.write(str(temp_tfidf).lstrip('[').rstrip(']'))
            file_tfidf.write('\n')
    file_tf.close()
    file_tfidf.close()
    end = time.clock()
    print "生成svm向量用时: %f s" % (end-start)
    return
def Svm_result(Dic_test):
    file_tf=open('feature_tf.txt','r')
    file_tfidf=open('feature_tfidf.txt','r')
    vector_tf = []    #基于tf的特征向量
    vector_tfidf = []    #基于tfidf的特征向量
    class_label = []  #svm训练用的标签
    max_min = []  #训练的最大最小值。
    total_tf = file_tf.readlines()
    total_tfidf = file_tfidf.readlines()
    #生成特征向量的类别标签
    path_now = os.getcwd()
    path_train = path_now + '\\'+'training'
    for file_name in os.listdir(path_train):
        text_path = path_train + '\\' + file_name
        for text_name in os.listdir(text_path):
            class_label.append(file_name)
    print class_label
    #生成基于词频的特征向量
    for word_tf in total_tf:
        word_tf = word_tf.strip('\n')
        tf_list = word_tf.split(', ')
        tf_list= map(int, tf_list) #将字符串转为整形
        vector_tf.append(tf_list)
    #生成基于词频逆文档的特征向量
    for word_tfidf in total_tfidf:
        word_tfidf = word_tfidf.strip('\n')
        tfidf_list = word_tfidf.split(', ')
        tfidf_list= map(float, tfidf_list)
        vector_tfidf.append(tfidf_list)
    file_tf.close()
    file_tfidf.close()
    (vector_tfidf,class_label,max_min) = normal_shuffle(vector_tfidf,class_label)
    print vector_tfidf[1000]
    print class_label[1000]
    clf = svm.SVC(C=32.0, cache_size=200, class_weight=None, coef0=0.0, degree=3,gamma=0.0,
        kernel='rbf', max_iter=-1, probability=False, random_state=None,shrinking=True,
        tol=0.001, verbose=False)
    print class_label
    clf.fit(vector_tfidf,class_label)
    print clf.predict(vector_tfidf[1000])
    return
def normal_shuffle(Vector,labels):#进行归一化和洗牌
    vec_len = len(Vector)
    fea_len = len(Vector[0])
    max_min = []
    print vec_len,fea_len
    #向量归一化
    for i in range(fea_len):
        list_column = []
        for j in range(vec_len):
            list_column.append(Vector[j][i])
        max_value = max(list_column)
        min_value = min(list_column)
        if max_value != 0:
            for k in range(vec_len):
                Vector[k][i] = round((Vector[k][i]-min_value)*1.0/(max_value-min_value),3)
        max_min.append([max_value,min_value])
    #进行洗牌
    for index in range(vec_len):
        Vector[index].append(labels[index])
    random.shuffle(Vector)
    label_new = []
    for index in range(vec_len):
        label_new.append(Vector[index].pop())
    return (Vector,label_new,max_min)
if __name__=='__main__':
    Dic_train = {}
    Dic_test = {}
    Dic_train = readfile('training')
    Dic_test = readfile('test')
    Svm_result(Dic_test)