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
from sklearn import tree


class adaboostMM:
    def __init__(self, rounds = 5):
        self.T = rounds
        self.wlearner = []
        self.alpha = np.zeros(rounds)
   
    def fit(self, X,Y):
        k = np.unique(Y)
        print k
        m = len(X)
        f = np.zeros((m, len(k)))
        COST = np.zeros((m, len(k)))
        weight=np.ones(m)
        print len(weight)
        weight=weight/float(m)

        
        for t in range(self.T):
            for i in range(m):
                for l in range(len(k)):
                    COST[i,l]=np.exp(f[i,l]-f[i,Y[i]])

            COST[np.array(range(m)), Y] = 0
            d_sum = np.sum(COST, axis = 1)
            COST[np.array(range(m)), Y] = -d_sum
            
            print 'weight is ',weight[1:50]
            min_weight=np.amin(weight)
            print min_weight
            weight=weight+min_weight;
            dc=tree.DecisionTreeClassifier(max_depth=6)
            temp_wlearner=dc.fit(X, Y, sample_weight=weight)
            self.wlearner.append(temp_wlearner)
            htx=temp_wlearner.predict(X)
            print htx[1:100], Y[1:100]
            print 'current alpha is ', t
            print 'accu is ', sum(htx==Y)/float(len(X))
       
            
            
            delta = np.sum(COST[np.array(range(m)), np.array(htx)])/(np.sum(d_sum))
            self.alpha[t] = 0.5 * np.log(1.0 * (1 + delta) / (1 - delta))
            print 'ALPHA ',self.alpha[t]
            weight=self.cost2weight(COST,htx)

            for i in range(m):
                for l in range(len(k)):
                    f[i,l] = f[i,l] + self.alpha[t] * (htx[i]==(l))
            
        
   
    

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
            X.append(np.array(ss))

        return X,y


    def ada_classifier(self, X, T):
        result=0
        ft = np.zeros((len(X), 10))
        #print len(examples)
        for t in range(T):
            htx = np.zeros_like(ft)
            temp_htx=self.wlearner[t].predict(X)
            htx[np.array(range(len(X))), np.array(temp_htx)] = self.alpha[t] 
            ft +=  htx
        final_pred = np.argmax(ft, axis = 1)
        print 'the final predict is ',type(final_pred[1:10])
        return final_pred+1

    

            
if __name__ == '__main__':
    train_acc=[]
    test_acc=[]
    adaMM_train=[]
    adaMM_test=[]
    T=1
    adaboost=adaboostMM(int(T))
    path_train='../../data/vw_multiclass.train' 
    path_test='../../data/vw_multiclass.test'

    X_train, Y_train = adaboost.C45_read_file('/Users/liguifan/Documents/nnet/data/vw_multiclass.test')
    X_test, Y_test = adaboost.C45_read_file('/Users/liguifan/Documents/nnet/data/vw_multiclass.test')
    # MNIST_train, Y_train, MNIST_test, Y_test=adaboost.read_MnistFile(path_train,path_test)
    adaboost.fit(X_train, Y_train)


    '''
    for t in range(T):
        train_htx_tmp=adaboost.wlearner[t].predict(csoaa_train)
        test_htx_tmp=adaboost.wlearner[t].predict(csoaa_test)
        train_htx=[int(i) for i in train_htx_tmp]
        test_htx=[int(i) for i in test_htx_tmp]
        train_acc.append(float(sum(train_htx==(Y_train+1)))/len(Y_train))
        test_acc.append(float(sum(test_htx==(Y_test+1)))/len(Y_test))
    ''' 
    
    t=T
    train_pred_label=adaboost.ada_classifier(X_train,t)
    test_pred_label=adaboost.ada_classifier(X_test,t)

    '''
    test_pred_label=adaboost.ada_classifier(csoaa_test,t)
    adaMM_train.append(float(sum(train_pred_label==(Y_train+1)))/len(train_pred_label))
    adaMM_test.append(float(sum(test_pred_label==(Y_test+1)))/len(test_pred_label))
    '''
    print 'train accuracy is ', float(sum(train_pred_label==(np.array(Y_train)+1)))/len(train_pred_label)
    print 'test accuracy is ', float(sum(test_pred_label==(np.array(Y_test)+1)))/len(test_pred_label)



    '''
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
    #plt.axis([40, 160, 0, 0.03])
    filename='plot1'
    fig.savefig(filename)
    ''' 




    


