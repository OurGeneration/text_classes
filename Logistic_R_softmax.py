#coding=utf-8
from scipy import sparse,io
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.datasets import load_files
from sklearn.cross_validation import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_selection import SelectKBest, chi2
from sklearn import metrics
from numpy import *
import warnings
warnings.filterwarnings("ignore")

#sigmoid函数定义
def sigmoid(wx):
    return 1.0/(1+exp(-wx))

#每个类别输出概率为prob[iClassNum]
def getPi(X,W,c):
    prob = []
    for i in range(c):
        prob.append(float(exp(W[i]*X))/float(1+sum(exp(W*X))))
    return prob

#随机梯度下降
def stocGradAscent(maxI,dataMatrix,classLabels,c):
    m,n = shape(dataMatrix)
    alpha = 0.5#学习率\梯度上升步长
    weights = mat(ones((c,n)))
    for k in range(maxI):
        #每个样本
        dCost = 0.0
        for i in range(m):
            #计算样例i的每个类别输出概率为prob[iClassNum]
            prob = getPi(dataMatrix[i].transpose(),weights,c)
            probi = prob.index(max(prob))
            #print probi,
            cc = classLabels[i]
            #print cc
            for cl in range(c):
                if cc == cl:
                    weights[cl] = weights[cl] - alpha*dataMatrix[i]*(prob[cl]-1)
                    dCost -= log(prob[cl])
                else:
                    weights[cl] = weights[cl] - alpha*dataMatrix[i]*prob[cl]
                    dCost -= log(1.0 - prob[cl])
        print 'id=', k
        ,
        print 'cost=', dCost/m
    return weights

def get_weights(dataMatrix,classLabels,maxl,matfilename):
    m,vectornum = shape(dataMatrix)
    print m,vectornum
    #第一列x0
    a = ones(m)
    vectormat = mat(c_[a,dataMatrix])
    weights = stocGradAscent(maxl, vectormat, classLabels, 10)
    data = {}
    data['weights'] = weights
    io.savemat(matfilename, data)
    return weights

def calculate_result(actual,pred):
    m_precision = metrics.precision_score(actual,pred)
    m_recall = metrics.recall_score(actual,pred)
    m_acc = metrics.accuracy_score(actual,pred)
    print 'predict info:'
    print 'accuracy:{0:.3f}'.format(m_acc)
    print 'precision:{0:.3f}'.format(m_precision)
    print 'recall:{0:0.3f}'.format(m_recall)
    print 'f1-score:{0:.3f}'.format(metrics.f1_score(actual,pred))

if __name__ == '__main__': 
    #load datasets
    doc_train = load_files('training')
    doc_test = load_files('test')
    #TF-IDF特征（词频）
    count_vec = TfidfVectorizer(min_df=1,decode_error='replace')   
    #Bool型特征（one-hot）
    #count_vec = CountVectorizer(binary = True,decode_error='replace')
    doc_train_bool = count_vec.fit_transform(doc_train.data)
    doc_test_bool = count_vec.transform(doc_test.data)
    train = doc_train_bool.toarray()
    test = doc_test_bool.toarray()
    print 'load finished'
    #训练权值
    weights = get_weights(train, doc_train.target,50,'weight.mat')
    data1 = io.loadmat('weight.mat')
    weights = data1['weights']

    m,vectornum = shape(test)
    a = ones(m)
    test = mat(c_[a,test])
    #预测所有测试集
    predicted = []
    for i in xrange(shape(test)[0]):
        probi = getPi(test[i].transpose(), weights, 10)
        x = probi.index(max(probi))
        predicted.append(x)
    
    calculate_result(doc_test.target,predicted)
    '''
    #调用sklearn包完成分类
    classifier = LogisticRegression()#LR 参数默认
    classifier.fit(vectormat, labeled_names)  #训练数据，无返回值
    pred = classifier.predict(vectormat1)
    calculate_result(labeled_names1,pred)
    '''
    c = zeros((10,10), dtype=int)
    for i in range(len(predicted)):
        c[doc_test.target[i]][predicted[i]-1] = c[doc_test.target[i]][predicted[i]-1] + 1
    print c