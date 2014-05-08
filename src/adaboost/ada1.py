'''
    adaboost.MM implementation.
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


class adaboostMM:
    def __init__(self, rounds = 5):
        self.T = rounds
        self.wlearner = []
        self.alpha = np.zeros(rounds)
        self.train_err=[]
    
    
    '''MNIST_DATA is a list of strings'''
    def fit(self, MNIST_DATA,Y):
        k = np.unique(Y)
        print k
        m = np.size(MNIST_DATA)
        f = np.zeros((m, len(k)))
        COST = np.zeros((m, len(k)))
        #ff=open('../../data/train_MM_2','w')
        for t in range(self.T):
            '''choose cost matrix C'''
            # set values where l != yi
            #COST = np.exp(f - np.choose(Y, f.T)[:, np.newaxis])
            
            for i in range(m):
                for l in range(len(k)):
                    COST[i,l]=np.exp(f[i,l]-f[i,Y[i]])
            

            # set values where l == yi
            COST[np.array(range(m)), Y] = 0
            d_sum = np.sum(COST, axis = 1)
            COST[np.array(range(m)), Y] = -d_sum
          
            csoaa_data=self.transform(COST,MNIST_DATA)
            name='cache_m'+str(t)
            self.wlearner.append(weak_learner('ML', name ,1, True, True, False))
            temp_htx=self.wlearner[-1].train(csoaa_data).predict(csoaa_data)
            htx=[int(i) for i in temp_htx]
            delta = -np.sum(COST[np.array(range(m)), np.array(htx)-1])/(np.sum(d_sum))
            self.alpha[t] = 0.5 * np.log(1.0 * (1 + delta) / (1 - delta))
            print 'ALPHA ',self.alpha[t]

            ind_vec_htx = np.zeros_like(f) 
            ind_vec_htx[np.array(range(m)), np.array(htx)-1] = self.alpha[t]
            f += ind_vec_htx
            
            print 'CURRENT ACCURACY FOR '+ `t`+' iteration is: ', float(sum(htx==(Y+1)))/m
            self.train_err.append(float(sum(htx==(Y+1)))/m)
    '''vw_mnist is a list type and COST_MATRIX is a ndarray type'''
    def transform(self, COST_MATRIX, vw_mnist):
        n_samples, n_features = np.shape(COST_MATRIX)
        result = []
        for i in range(n_samples):
            tuple_exampe=vw_mnist[i].split('| ')
            feature_value=tuple_exampe[1]
            vw_csoaa_example=' '.join([' '.join([str(j+1)+':'+`float(COST_MATRIX[i,j])` for j in range(n_features)]),'|',feature_value])
            #print vw_csoaa_example
            result.append(vw_csoaa_example)

        return result
    

    def single_predict(self, instance):
        instances = []
        with self.model.predicting():
            self.model.push_instance(instance)
            instances.append(instance)
        prediction = list(self.model.read_predictions_())
        return  prediction

    '''
    def read_MnistFile(self, file_path):
        examples=open(file_path,"r")
        mnist_after=[]
        examples_no=0
        for example in examples:
            mnist_after.append(example)
            examples_no+=1
        examples.close()

        examples=open(file_path,"r")
        class_set=np.zeros(examples_no,dtype=int)
        m=0
        for ex in examples:
            class_set[m]= ord(ex[0])-48
            m+=1
        examples.close()
        return (mnist_after,class_set)
    '''


    def read_MnistFile(self, file_path_train,file_path_test):
        train_examples=open(file_path_train,"r")
        test_examples=open(file_path_test,"r")
        train_mnist_after=[]
        test_mnist_after=[]
        train_class_set=[]
        test_class_set=[]

        for example in train_examples:
            train_mnist_after.append(example)
            train_class_set.append(int(ord(example[0])-48))
        train_examples.close()

        for example in test_examples:
            test_mnist_after.append(example)
            test_class_set.append(int(ord(example[0])-48))
        test_examples.close()
        

        return (train_mnist_after, np.array(train_class_set), test_mnist_after, np.array(test_class_set))



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
    T=500
    adaboost=adaboostMM(int(T))
    path_train='../../data/vw_multiclass.train' 
    path_test='../../data/vw_multiclass.test'


    MNIST_train, Y_train, MNIST_test, Y_test=adaboost.read_MnistFile(path_train,path_test)
    adaboost.fit(MNIST_train,Y_train)

    csoaa_train,csoaa_test = adaboost.test_process(MNIST_train,MNIST_test)

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
    train_pred_label=adaboost.ada_classifier(csoaa_train,t)
    test_pred_label=adaboost.ada_classifier(csoaa_test,t)
    adaMM_train.append(float(sum(train_pred_label==(Y_train+1)))/len(train_pred_label))
    adaMM_test.append(float(sum(test_pred_label==(Y_test+1)))/len(test_pred_label))
    print 'train accuracy is ', float(sum(train_pred_label==(Y_train+1)))/len(train_pred_label)
    print 'test accuracy is ', float(sum(test_pred_label==(Y_test+1)))/len(test_pred_label)




    print [adaboost.alpha[i] for i in range(int(T))]
    print 'train acc is ',adaMM_train
    print 'test acc is ', adaMM_test

    alpha_train=open('../adaResult/alpha','w') 
    train_individual=open('../adaResult/train_individual','w') 
    for i in range(int(T)):
        alpha_train.write(str(adaboost.alpha[i])+'\n')
        train_individual.write(str(adaboost.train_err[i])+'\n')
    alpha_train.close()
    train_individual.close()

    fig = plt.figure()
    iters = T
    x_axis = range(1,iters+1)
    
    plt.plot(x_axis, adaboost.alpha, 'k--', label = 'train_accuracy')

    '''
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
     




    


