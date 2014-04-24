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


class adaboostMM:
    def __init__(self, rounds = 5):
        self.T = rounds
        self.wlearner = []
        self.alpha = np.zeros(rounds)
    
    
    '''MNIST_DATA is a list of strings'''
    def fit(self, MNIST_DATA,Y):


        k = np.unique(Y)
        m = np.size(MNIST_DATA)
   
        '''In our case, the k is 10 for MNIST data set'''
        f = np.zeros((m, len(k)))
        COST = np.zeros((m, len(k)))
        #vw_cost is the cost matrix in vowpal wabbit conpatibel version
        


        for t in range(self.T):
            '''choose cost matrix C'''
            # set values where l != yi
            #C = np.exp(f - np.choose(Y, f.T)[:, np.newaxis])
            
            for i in range(m):
                for l in range(len(k)):
                    COST[i,l]=np.exp(f[i,l]-f[i,Y[i]])
            

            # set values where l == yi
            COST[np.array(range(m)), Y] = 0
            d_sum = np.sum(COST, axis = 1)
            COST[np.array(range(m)), Y] = -d_sum
           
            print 'MIN value in cost matrix is ', np.amin(COST)

            #peakIndexTuple = np.unravel_index(np.argmax(COST), COST.shape)

            #for x in csoaa_data:
            #     tempfile.write(str(x))
            # break

            '''write the Cost matrix into file'''

            min_element=np.amin(COST)
            print 'MIN ELEMENT IS ',min_element
            #csoaa is a list of strings with the format vw takes 
            csoaa_data=self.transform(COST,MNIST_DATA,min_element)
          
            #call vowpal wabbit for training a weak classifier.
            name='cache_m'+str(t)
            self.wlearner.append(weak_learner('ML', name ,10))
            temp_htx=self.wlearner[-1].train(csoaa_data).predict(csoaa_data)
    
            #htx is an array of prediction across the whole data in integer format
            htx=[int(i) for i in temp_htx]

            #calculate delta using the predicions, cost matrix and f
            delta = -np.sum(COST[np.array(range(m)), np.array(htx)-1])/(np.sum(d_sum))
            
            #calculate alpha with natural log
            self.alpha[t] = 0.5 * np.log(1.0 * (1 + delta) / (1 - delta))
            print 'ALPHA ',self.alpha[t]

            #update f matrix
            for i in range(m):
                for l in range(len(k)):
                    f[i,l] = f[i,l] + self.alpha[t] * (htx[i]==(l+1))

            '''
            ind_vec_htx = np.zeros_like(f) 
            ind_vec_htx[np.array(range(m)), np.array(htx)-1] = self.alpha[t]
            print 'ALPHA', self.alpha[t]
            f += ind_vec_htx
            '''
            
            print 'CURRENT ACCURACY FOR '+ `t`+' iteration is: ', float(sum(htx==(Y+1)))/m
    
    '''vw_mnist is a list type and COST_MATRIX is a ndarray type'''
    def transform(self, COST_MATRIX, vw_mnist, min_element):
        n_samples, n_features = np.shape(COST_MATRIX)
        result = []
        for i in range(n_samples):
            tuple_exampe=vw_mnist[i].split('| ')
            feature_value=tuple_exampe[1]
            vw_csoaa_example=' '.join([' '.join([str(j+1)+':'+`float(COST_MATRIX[i,j]+100.0)` for j in range(n_features)]),'|',feature_value])
            result.append(vw_csoaa_example)

        return result
    

    def single_predict(self, instance):
        instances = []
        with self.model.predicting():
            self.model.push_instance(instance)
            instances.append(instance)
        prediction = list(self.model.read_predictions_())
        return  prediction



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


    '''For this case, we have 10 classes <1...10>'''
    def ada_classifier(self, examples):
        result=0
        ft = np.zeros((len(examples), 10))
        print len(examples)
        for t in range(self.T):
            htx = np.zeros_like(ft)
            temp_htx=self.wlearner[t].predict(examples)
            index=[int(i) for i in temp_htx]
            htx[np.array(range(len(examples))), np.array(index)-1] = self.alpha[t] 
            ft +=  htx
        final_pred = np.argmax(ft, axis = 1)
        return final_pred+1

    def test_process(self, MNIST):
        examples=[]
        for example in MNIST:
            tuple_exampe=example.split('| ')
            feature_value=tuple_exampe[1]
            examples.append('1:1 2:1 3:1 4:1 5:1 6:1 7:1 8:1 9:1 10:1 | '+feature_value)
        return examples
            





    


