#encoding:utf-8  实现文本分类
import sys
import os
import re
import time
import math
import random
import jieba
import jieba.posseg as pseg
from numpy import *
from sklearn import preprocessing
from sklearn import svm
from sklearn.cross_validation import train_test_split
import numpy as np
import scipy as sp
import matplotlib.pyplot as plt
from nltk.corpus import stopwords
from operator import itemgetter
from collections import OrderedDict,Counter
from collections import defaultdict,Counter
stopset = set(stopwords.words('english'))  #英文停用词
def pretreatment(str_txt):#对读取的英文做预处理
    str_txt = str_txt.strip()
    str_txt = str_txt.lower()
    str_txt = str_txt.replace('\n',' ')
    str_txt = re.compile('-|_|/').sub(' ',str_txt)
    str_txt = re.compile('\d+.\d*').sub(' ',str_txt)
    str_txt = re.compile('\d+').sub(' ',str_txt)
    str_txt = re.compile('<[^>]+>').sub(' ',str_txt)
    str_txt = re.compile('\(.*?\)').sub(' ',str_txt)
    str_txt = re.compile('\s{2,}').sub(' ',str_txt)
    str_txt = re.sub('[’!"#$%&\'()*+,-./:;<=>?@[\\]^_`{|}~]+',' ',str_txt)
    return str_txt
def readsword():  #添加停用词词表
    stop = []
    readsw = open("stopwords.txt",'r').readlines()
    for word in readsw:
        stop.append(word.strip('\n'))
    return stop
def stopwords(sen_data): #传入的为字符串,去除停用词
    str_process = ""
    stopset1 = readsword()
    sen_list = sen_data.split()
    list_process = [word for word in sen_list if word not in stopset and word not in stopset1]
    str_process = ' '.join(list_process)
    #print str_process
    return str_process
def readfile(type_read):
    #使用os.listdir获取路径
    path_now = os.getcwd()
    path_train = path_now + '\\'+type_read
    Data_dic = {}
    for file_name in os.listdir(path_train):
        class_name = file_name
        print "读取",type_read,"中的类别",class_name
        text_path = path_train + '\\' + file_name
        for text_name in os.listdir(text_path):
            final_path = text_path + '\\' + text_name  #text_name
            all_text = open(final_path).read( )     # 文本文件中的所有文本
            text_process = pretreatment(all_text)
            Data_dic.setdefault(class_name,[]).append(stopwords(text_process))
    return Data_dic
def write_table(Dic_data):  #将构建的词表写入文本，大概需要20分钟左右
    dicfile=open('word_table.txt','w')
    for key,num_word in Dic_data.items():
        #print key,num_word
        dicfile.write(str(key)+' '+str(num_word).lstrip('[').rstrip(']'))
        dicfile.write('\n')
    dicfile.close()
    return
def read_table(): #读入文本文件中的词表，存入字典
    table = {}
    file_table = open('word_table.txt','r')
    words_table = file_table.readlines()
    for word_num in words_table:
        temp_list = word_num.strip('\n').split(' ',1)
        word = temp_list[0]
        num_list = temp_list[1].split(', ')
        num_list= map(int, num_list) #将字符串转为整形
        table.setdefault(word,[]).append(num_list)
    file_table.close()
    return table
def get_word(Dic_data,num_total):  #获取前num_total个特征
    top_fea = []
    count = 0
    for k,v in Dic_data.items():
        if count < num_total:
            fea_name = k#.replace("CHI",'')
            top_fea.append(fea_name)
            count = count + 1
    return top_fea

def Cal_chi(Dan_word,Dic_data,class_name):#计算每一个类别的chi排序,取前1000个。
    num_A,num_B,num_C,num_D = 0,0,0,0
    labels = ['acq','corn','crude','earn','grain','interest','money-fx','ship','trade','wheat']
    word_table = read_table() #从文本中读取词表
    CHI_result = {}
    label_index = labels.index(class_name)
    for chi_word in Dan_word:
        num_A,num_B,num_C,num_D = 0,0,0,0
        label_list = word_table[chi_word]
        num_A = label_list[0][label_index]
        num_C = len(Dic_data[class_name]) - num_A
        for num_rest in label_list[0]:
            if label_list[0].index(num_rest) != label_index:
                num_B = num_B + num_rest
        for label in labels:
            if labels.index(label) != label_index:
                num_D = num_D + len(Dic_data[label])
        num_D = num_D - num_B
        numerator = (num_A*num_D-num_B*num_C)  #计算分子
        denominator = (num_A + num_B)*(num_C + num_D)  #计算分母
        CHI_word =1.0*numerator*numerator/(1.0*denominator)
        CHI_result[chi_word] = round(CHI_word,2)   #保留小数点后两位
    #按照字典中的CHI值进行降序排序
    CHI_result = OrderedDict(sorted(CHI_result.iteritems(), key=itemgetter(1), reverse=True))
    each_1000 = get_word(CHI_result,1000) #对每个类别取前1000个作为特征
    #print each_1000
    return each_1000
def CHI_feature(Dic_data): #用CHI进行特征选择
    labels = ['acq','corn','crude','earn','grain','interest','money-fx','ship','trade','wheat']
    Dan_words = []
    all_fea = []
    for label in labels:
        Dan_words = []
        for sen in Dic_data[label]:
            Dan_words = Dan_words + sen.split()
        dan_words = list(set(Dan_words))#列表元素去重
        all_fea.append(Cal_chi(dan_words,Dic_data,label))
    file_path = open('feature.txt','w')
    for i in range(len(all_fea)):  #写文件
        file_path.write(str(i)+'\n')
        for j in all_fea[i]:
            file_path.write(str(j)+'\n')
    file_path.close()
    #print all_fea
    return all_fea

def IG_feature(Dic_data): #用IG信息增益进行特征选择
    all_words = []
    for class_name in Dic_data:
        for sen in Dic_data[class_name]:
            all_words = all_words + sen.split()
    words = list(set(all_words))#列表元素去重
    return
def build_table(Dic_data): #建立每个词的在每个类中的数量表
    labels = ['acq','corn','crude','earn','grain','interest','money-fx','ship','trade','wheat']
    word_table = {}
    all_words = []
    list_num = []
    #words = ['prohibited', 'offset', 'mir', 'java', 'dillon', 'thasos', 'refuge', 'walahtam', 'barreto']
    for class_name in Dic_data:
        for sen in Dic_data[class_name]:
            all_words = all_words + sen.split()
    words = list(set(all_words))#列表元素去重
    start = time.clock()
    #print words
    for word in words:
        list_num = []
        for label in labels: #计算每个单词在每个类别中的个数
            count_label = 0
            for label_sen in Dic_data[label]:
                if word in set(label_sen.split()):
                    count_label = count_label + 1
            list_num.append(count_label)
        word_table.setdefault(word,[]).append(list_num)
    #print word_table
    end = time.clock()
    print "run: %f s" % (end-start)
    write_table(word_table)
    return
if __name__=='__main__':
    Dic_train = {}
    Dic_test = {}
    Dic_train = readfile('training')
    Dic_test = readfile('test')