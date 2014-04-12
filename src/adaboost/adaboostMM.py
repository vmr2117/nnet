'''
    adaboost.MM implementation.
    '''

import numpy as np
from vowpal_porpoise import VW
import sys
import os
import tempfile
import commands
import itertools


class Instance(object):
    def __init__(self, vw_fmt):
        self.vw_fmt = vw_fmt
      
    def featurize(self):
        raise Exception('Not yet implemented: Instance.featurize')

    def __repr__(self):
        return self.vw_fmt

class adaboostMM:
    def __init__(self,  moniker, path, rounds = 5):
        self.T = rounds
        self.moniker=moniker
        self.wlearner = []
        self.alpha = np.zeros(rounds)
        self.model=VW(moniker=moniker, name= 'cache_d', passes=5 , csoaa=10)
    
    
    '''MNIST_DATA is a list of strings'''
    def fit(self, MNIST_DATA,Y):
        k = np.unique(Y)
        m = np.size(MNIST_DATA)
   
        '''In our case, the k is 10 for MNIST data set'''
        f = np.zeros((m, len(k)+1))
        C = np.zeros((m, len(k)+1))



        
        for t in range(self.T):
     
            '''choose cost matrix C'''
            # set values where l != yi
            print 't value is ',t
            C = np.exp(f - np.choose(Y, f.T)[:, np.newaxis])
           
            # set values where l == yi
            C[np.array(range(m)), Y] = 0
        
            d_sum = -np.sum(C, axis = 1)
            C[np.array(range(m)), Y] = d_sum

            #vw_cost is the cost matrix in vowpal wabbit conpatibel version
            csoaa_data=self.transform(C,MNIST_DATA)


            #for x in csoaa_data:
            #     tempfile.write(str(x))
            # break
            
            #csoaa should be a list of Instance rather than a list of strings
            #call vowpal wabbit for training a weak classifier.
            
            #self.wlearner.append(c)
            self.train(csoaa_data)
            #_, prediction_file = tempfile.mkstemp(dir='.', prefix=self.model.get_prediction_file())


            temp_htx = self.predict(csoaa_data)
            htx=[int(i) for i in temp_htx]
            print 'value of d_sum is   ', np.sum(d_sum)
            #predicion on train set
            #htx is an array of prediction across the whole data
            
            #_,htx=self.wlearner[t].predict(csoaa_data)
            

            #theta = weaklearner parameters
            #htx - predicted y and it is an array of predicted values
            
            #calculate delta using the predicions, cost matrix and f
            print C
            print np.array(range(m))
            delta = -np.sum(C[np.array(range(m)), np.array(htx)])/np.sum(d_sum)
            print delta
            #calculate alpha
            self.alpha[t] = 0.5 * np.log(1.0 * (1 + delta) / (1 - delta))
            #update f matrix
            Y=','.join(Y)
            print  Y
            f = f + self.alpha[t] * (htx == Y)

    #output final classifier weights.
    
    
    
    '''vw_mnist is a list type and COST_MATRIX is a ndarray type'''
    def transform(self, COST_MATRIX, vw_mnist):
        n_samples, n_features = np.shape(COST_MATRIX)
        result = []
        for i in range(n_samples):
            tuple_exampe=vw_mnist[i].split('| ')
            feature_value=tuple_exampe[1]
            vw_csoaa_example=' '.join([' '.join([str(j)+':'+`COST_MATRIX[i,j]` for j in range(1,n_features) if COST_MATRIX[i,j] != 0]),'|',feature_value])
            #Instance(vw_csoaa_example)
            result.append(vw_csoaa_example)

        return result
    
    def train(self, instance_stream):
        with self.model.training():
            seen=0
            for instance in instance_stream:
                self.model.push_instance(instance)
                seen+=1
                if seen % 1000 ==0:
                    print 'setreamed %d instances already' % seen
            print 'streaming finished'
        print '%s: trained on %d data points' % (self.moniker, seen)
        return self
    
    
    def predict(self, instance_stream):
        print '%s: predicting' % self.moniker
        instances = []
        seen=0
        
        with self.model.predicting():
            seen = 0
            for instance in instance_stream:
                self.model.push_instance(instance)
                instances.append(instance)
                seen += 1
        print '%s: predicted for %d data points' % (self.moniker, seen)
        predictions = list(self.model.read_predictions_())
        if seen != len(predictions)-1:
           raise Exception("Number of labels and predictions do not match!  (%d vs %d)" % \
                            (seen, len(predictions)))
        print predictions[:len(predictions)-1]
        return  predictions[:len(predictions)-1]
    
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
            class_set[m]= ord(ex[0])-48 +1
            #print class_set[m]
            m+=1
            
        examples.close()
        return (mnist_after,class_set)



if __name__ == '__main__':


    '''    for (instance, prediction) in SimpleModel('example1').train(instances).predict(instances):
        print prediction, instance
    '''

    current_directory=os.getcwd()
    filename='validation_part_original'
    path=os.path.join(current_directory, filename)

    adaboost=adaboostMM('liguifan',path, )
    MNIST, Y=adaboost.read_MnistFile(path)
    adaboost.fit(MNIST,Y)

    '''MNIST_DATA=read_MnistFile(path)
    fit(MNIST_DATA)'''
