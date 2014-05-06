'''afd
'''
import numpy as np
import sys
import os
import tempfile
import commands
import itertools
from weak_learnerMM import weak_learner
import argparse
import pylab as plt
from sklearn.tree import DecisionTreeClassifier


class adaboostMM:
    def __init__(self, rounds = 5):
        self.T = rounds
        self.wlearner = []
        self.alpha = np.zeros(rounds)
   
    def fit(self, X,Y):
        k = np.unique(Y)
        print len(X)
        m = len(X)
        f = np.zeros((m, len(k)))
        COST = np.zeros((m, len(k)))
        weight=np.ones(m)
        print len(weight)
        weight=weight/float(m)
        wlearner=[]
        
        for t in range(self.T):
            for i in range(m):
                for l in range(len(k)):
                    COST[i,l]=np.exp(f[i,l]-f[i,Y[i]])

            COST[np.array(range(m)), Y] = 0
            d_sum = np.sum(COST, axis = 1)
            COST[np.array(range(m)), Y] = -d_sum
            
            print 'weight is ',weight[1:50]
            dc=DecisionTreeClassifier()
            temp_wlearner=dc.fit(X, Y, sample_weight=weight)
            wlearner.append(temp_wlearner)
            htx=temp_wlearner.predict(X)
            print htx[1:50]
            print 'accu is ', sum(htx==Y)/10000.0
       
            weight=self.cost2weight(COST,htx)
            
            delta = -np.sum(COST[np.array(range(m)), np.array(htx)-1])/(np.sum(d_sum))
            self.alpha[t] = 0.5 * np.log(1.0 * (1 + delta) / (1 - delta))
            print 'ALPHA ',self.alpha[t]

            for i in range(m):
                for l in range(len(k)):
                    f[i,l] = f[i,l] + self.alpha[t] * (htx[i]==(l+1))
            
            #print 'CURRENT ACCURACY FOR '+ `t`+' iteration is: ', float(sum(htx==(Y+1)))/m
    
   
    

    def cost2weight(self, COST,predict):
        return COST[np.array(range(len(predict))), np.array(predict)]

  
    def C45_read_file(self,path):
        X=[]
        y=[]
        file1=open(path,'r')
        for example in file1:
            y.append(int(ord(example[0]))-48)
            tuple_exampe=example.split(' | ')
            feature_value=tuple_exampe[1]
            mm=feature_value[:-2].split(' ')
            ss=[float(x) for x in mm] 
            X.append(ss)

        return X,y




    '''For this case, we have 10 classes <1...10>'''
    def ada_classifier(self, examples, T):
        result=0
        ft = np.zeros((len(examples), 10))
        #print len(examples)
        for t in range(T):
            htx = np.zeros_like(ft)
            temp_htx=self.wlearner[t].predict(examples)
            index=[int(i) for i in temp_htx]
            htx[np.array(range(len(examples))), np.array(index)-1] = self.alpha[t] 
            ft +=  htx
        final_pred = np.argmax(ft, axis = 1)
        return final_pred+1

    def test_process(self, MNIST_train, MNIST_test):
        train_examples=[]
        test_examples=[]

        for example in MNIST_train:
            tuple_exampe=example.split('| ')
            feature_value=tuple_exampe[1]
            train_examples.append('1:1 2:1 3:1 4:1 5:1 6:1 7:1 8:1 9:1 10:1 | '+feature_value)

        for example in MNIST_test:
            tuple_exampe=example.split('| ')
            feature_value=tuple_exampe[1]
            test_examples.append('1:1 2:1 3:1 4:1 5:1 6:1 7:1 8:1 9:1 10:1 | '+feature_value)

        return train_examples, test_examples


    def test_train_test_errro(self, training, test, train_labels, test_labels):
        train_error=[]
        test_error=[]
        for t in range(self.T):
            train_htx_temp=self.wlearner[t].predict(training)
            test_htx_temp=self.wlearner[t].predict(test)
            train_htx=[int(i) for i in train_htx_temp]
            test_htx=[int(i) for i in test_htx_temp]
            train_error.append(sum(train_htx==(train_labels+1))/len(train_htx))
            test_error.append(sum(test_htx==(test_labels+1))/len(test_htx))

        return train_error,test_error

            
if __name__ == '__main__':
    train_acc=[]
    test_acc=[]
    adaMM_train=[]
    adaMM_test=[]
    T=3
    adaboost=adaboostMM(int(T))
    path_train='../../data/vw_multiclass.train' 
    path_test='../../data/vw_multiclass.test'

    X, y = adaboost.C45_read_file('/Users/liguifan/Documents/nnet/data/niubi.test')
    # MNIST_train, Y_train, MNIST_test, Y_test=adaboost.read_MnistFile(path_train,path_test)
    adaboost.fit(X, y)

    csoaa_train,csoaa_test = adaboost.test_process(MNIST_train,MNIST_test)

    '''
    for t in range(T):
        train_htx_tmp=adaboost.wlearner[t].predict(csoaa_train)
        test_htx_tmp=adaboost.wlearner[t].predict(csoaa_test)
        train_htx=[int(i) for i in train_htx_tmp]
        test_htx=[int(i) for i in test_htx_tmp]
        train_acc.append(float(sum(train_htx==(Y_train+1)))/len(Y_train))
        test_acc.append(float(sum(test_htx==(Y_test+1)))/len(Y_test))
       
    
    t=T
    train_pred_label=adaboost.ada_classifier(csoaa_train,t)
    test_pred_label=adaboost.ada_classifier(csoaa_test,t)
    adaMM_train.append(float(sum(train_pred_label==(Y_train+1)))/len(train_pred_label))
    adaMM_test.append(float(sum(test_pred_label==(Y_test+1)))/len(test_pred_label))
    print 'train accuracy is ', float(sum(train_pred_label==(Y_train+1)))/len(train_pred_label)
    print 'test accuracy is ', float(sum(test_pred_label==(Y_test+1)))/len(test_pred_label)




    print [adaboost.alpha[i] for i in range(int(T))]
    print adaMM_train
    print adaMM_test

    fig = plt.figure()
    iters = len(adaMM_train)
    x_axis = range(1,iters+1)
    
    plt.plot(x_axis, adaMM_train, 'k--', label = 'train_accuracy')
    plt.plot(x_axis, adaMM_test, 'r--', label = 'test_accuracy')
    plt.legend()
    plt.xlabel('The # of weak learners')
    plt.ylabel('Accuracy')
    plt.title('AdaboostMM')
    plt.grid(True)
    plt.show()
    #plt.axis([40, 160, 0, 0.03])\
    filename='plot1'
    fig.savefig(filename)
    ''' 




    


