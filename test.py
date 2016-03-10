#encoding:utf-8  实现文本分类
import sys
import re
import os
import random
import jieba
import jieba.posseg as pseg
from numpy import *
from sklearn import svm
from sklearn.cross_validation import train_test_split
import numpy as np
import scipy as sp
s='{通配符}你好，今天开学了{通配符},你好'
print("s", s)
a1 = re.compile(r'\{.*?\}' )
d = a1.sub('',s)
print type(d)
print("d",d)

a1 = re.compile(r'\{[^}]*\}' )
d = a1.sub('',s)
print("d",d)
seg_list = jieba.cut("他来到了网易杭研大厦。")  # 默认是精确模式
print ", ".join(seg_list)
words = pseg.cut("他来到了网易杭研大厦。")
strword = ""
for w in words:
    strword = strword + w.word+' '
print strword
seg_list = jieba.cut_for_search("小明硕士毕业于中国科学院计算所，后在日本京都大学深造。")  # 搜索引擎模式
print ", ".join(seg_list)
'''corpus.append(index.rstrip('\n').rstrip())
    vectorizer=CountVectorizer()#该类会将文本中的词语转换为词频矩阵，矩阵元素a[i][j] 表示j词在i类文本下的词频
    transformer=TfidfTransformer()#该类会统计每个词语的tf-idf权值
    x = vectorizer.fit_transform(corpus)
    listname = vectorizer.get_feature_names()
    for name in listname:
        feature.write(name.encode('utf-8').strip().strip('\n'))
        feature.write('\n')
         print Dic['acq'][0]
    print Dic['acq'][1]
    print Dic['wheat'][0]
    print Dic['wheat'][1]'''
def fenci(Data_str):
    str_sen = ""
    seg_list = pseg.cut(Data_str) #默认是精确模式
    for w in seg_list:
        str_sen = str_sen + str(w.word)+'/'
    str_sen = str_sen.replace(' ','')
    str_sen = str_sen.replace('//',' ')
    str_sen = str_sen.replace('/',' ')
    str_sen = str_sen.strip()
    return str_sen
all_fea = [['shares', 'acquisition', 'acquire'],['asda','nihao'],['her','sheis']]

def readfile(type_read):
    #使用os.listdir获取路径
    path_now = os.getcwd()
    path_train = path_now + '\\'+type_read
    Data_dic = {}
    for file_name in os.listdir(path_train):
        count = 0
        class_name = file_name
        print class_name
        text_path = path_train + '\\' + file_name
        for text_name in os.listdir(text_path):
            count = count + 1
        print file_name,count
    return Data_dic
readfile('training')
def Cal_chi(Dan_word,Dic_data,class_name):#计算每一个类别的chi排序,取前1000个。时间较长，不可取！！！
    num_A,num_B,num_C,num_D = 0,0,0,0
    labels = ['acq','corn','crude','earn','grain','interest','money-fx','ship','trade','wheat']
    CHI_result = {}
    remain = [] #剩余的类别
    #print Dan_word
    for label in labels: #将剩余的类别的文档放列表
        if label != class_name:
            for rest in Dic_data[label]:
                remain.append(rest)
    for chi_word in Dan_word:   #计算词的chi值
        num_A,num_B,num_C,num_D = 0,0,0,0
        for sentence in Dic_data[class_name]:
            sen = sentence.split()
            if chi_word in sen:
                num_A = num_A + 1
            else :
                num_C = num_C + 1
        for rest_sen in remain:
            rest_sen = rest_sen.split()
            if chi_word in rest_sen:
                num_B = num_B + 1
            else :
                num_D = num_D + 1
        numerator = (num_A*num_D-num_B*num_C)  #计算分子
        denominator = (num_A + num_B)*(num_C + num_D)  #计算分母
        CHI_word =1.0*numerator*numerator/(1.0*denominator)
        CHI_result[chi_word] = round(CHI_word,2)   #保留小数点后两位
    CHI_result = OrderedDict(sorted(CHI_result.iteritems(), key=itemgetter(1), reverse=True))
    #print CHI_result
    #print len(CHI_result)
    each_1000 = get_word(CHI_result,1000) #对每个类别取前1000个作为特征
    return each_1000
def svm_test():
    X = [[0,0,0],[1,1,2],[2,5,7],[1,10,20]]
    y = ['0','1','2','3']
    clf = svm.SVC(C=1.0, cache_size=200, class_weight=None, coef0=0.0, degree=3,gamma='auto',
        kernel='rbf', max_iter=-1, probability=False, random_state=None,shrinking=True,
        tol=0.001, verbose=False)
    clf.fit(X, y)
    print clf.predict([1,8,13])
    return True
svm_test()
list1 = [[1,2,3,4],[2,3,4,5],[3,4,5,6],[4,5,6,7]]
random.shuffle(list1)
print list1
alist = [123, 'xyz', 'zara', 'abc']
print "A List : ", alist.pop()
print alist
print "B List : ", alist.pop(2)
print alist