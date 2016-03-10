#encoding:utf-8  实现文本分类之逻辑回归
import sys
import os
import re
import time
import math
import random
import copy as copylist
from numpy import *
import matplotlib.pyplot as plt
import numpy as np
import scipy as sp
from classes import readfile
def sigmoid(inX):
    return 1.0 / (1 + exp(-inX))
def normal_vector(Vector):
    vec_len = len(Vector)
    fea_len = len(Vector[0])
    for i in range(fea_len):
        list_column = []
        for j in range(vec_len):
            list_column.append(Vector[j][i])
        max_value = max(list_column)
        min_value = min(list_column)
        if max_value != 0 and max_value != min_value:
            for k in range(vec_len):
                Vector[k][i] = round((Vector[k][i]-min_value)*1.0/(max_value-min_value),3)
    return Vector

def loadDataSet(doc_c):
    labels = {'acq':0,'corn':1,'crude':2,'earn':3,'grain':4,'interest':5,'money-fx':6,'ship':7,'trade':8,'wheat':9}
    vector_tfidf = []    #用tfidf表示向量
    class_label = []  #训练文档的标签
    path_now = os.getcwd()
    if doc_c == 'train':
        file_tfidf=open('feature_tfidf.txt','r')
        path_current = path_now + '\\'+'training'
    elif doc_c == 'test':
        file_tfidf=open('Doctest_tfidf.txt','r')
        path_current = path_now + '\\'+'test'
    total_tfidf = file_tfidf.readlines()
    for file_name in os.listdir(path_current):
        text_path = path_current + '\\' + file_name
        for text_name in os.listdir(text_path):
            class_label.append(labels[file_name])
    #生成基于词频逆文档的文档向量
    for word_tfidf in total_tfidf:
        tfidf_list = word_tfidf.strip('\n').split(', ')
        tfidf_list.insert(0,1)
        tfidf_list= map(float, tfidf_list)
        vector_tfidf.append(tfidf_list)
    #将类别标签加入文本向量的末尾
    for index in range(len(class_label)):
        vector_tfidf[index].append(class_label[index])
    return vector_tfidf

def dataPro(vector_P,current_P):
    label_new = []
    random.shuffle(vector_P)
    for index in range(len(vector_P)):
        if vector_P[index].pop() == current_P:
            label_new.append(1)
        else:
            label_new.append(0)
    return vector_P,label_new
def initial_w(num): #初始化回归系数
    initial_weight = []
    for i in range(num):
        initial_weight.append(random.uniform(0, 0.1))
    return array(initial_weight)
def cost_sample(w_temp,sample_data,sample_c):
    cost = 0.0
    cost = cost + sample_c*dot(w_temp,array(sample_data))
    cost = cost - math.log((1.0+exp(dot(w_temp,array(sample_data)))),math.e)
    return  cost
#改进的随机梯度下降算法
def stocGrad_Descent(datamat,classLabels,numIter):#参数为训练数据集和当前处理的类
    current_cost = 0
    print len(datamat),datamat[0]
    row_num,column_num = shape(array(datamat))
    weights = initial_w(column_num) #生成一个num维行向量
    for j in range(numIter):
        dataIndex = range(row_num)
        for index in range(row_num):
            alpha = 4 / (1.0 + j + index) + 0.05
            randIndex = int(random.uniform(0, len(dataIndex)-1))
            h = sigmoid(sum(array(datamat[randIndex])*weights))
            error = (h - classLabels[randIndex])
            weights = weights - alpha*error*array(datamat[randIndex])
            current_cost += cost_sample(weights,datamat[randIndex],classLabels[randIndex])
            del(dataIndex[randIndex])
        current_cost = current_cost*(-1)/row_num*1.0
        #print 'ID=',j,'cost=',current_cost
    return weights
#获得测试集样例的类别
def getPi(X,W):
    prob = []
    for i in range(10):
        for j in W:
             prob.append( sigmoid( dot( array(X),j ) ) )
    return prob.index(max(prob))
#逻辑胡桂预测，参数为类别的个数
def Logistic_R(class_num):
    Weights,vector_test,prob_label,testlabel = [],[],[],[]
    correct_num = 0
    for c in range(class_num):
        print '处理第',c+1,'个类'
        vector_train,labelmat = [],[]
        vector_train,labelmat = dataPro(loadDataSet('train'),c)
        vector_train = normal_vector(vector_train)
        Weights.append(stocGrad_Descent(vector_train,labelmat,3))
    vector_test = loadDataSet('test')
    for index in range(len(vector_test)):
        testlabel.append(vector_test[index].pop())
        prob_label.append( getPi(vector_test[index],Weights) )
    for i in range(len(testlabel)):
        if testlabel[i] == prob_label[i]:
            correct_num += 1
    Precision = correct_num*1.0/len(testlabel)
    print "逻辑回归的准确率为：",Precision
    return
if __name__=='__main__':
    Logistic_R(10)