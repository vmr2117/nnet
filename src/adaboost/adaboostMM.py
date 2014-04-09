'''
    adaboost.MM implementation.
    '''

import numpy as np
from vowpal_porpoise import VW
import sys

class Instance(object):
    def __init__(self, vw_fmt):
        self.vw_fmt = vw_fmt
      
    def featurize(self):
        raise Exception('Not yet implemented: Instance.featurize')

    def __repr__(self):
        return self.vw_fmt

'''
class SimpleInstance(Instance):
    def featurize(self):
        return {'a': self.raw_features}
'''

class adaboostMM:
    def __init__(self,  moniker, path, rounds = 5):
        self.T = rounds
        self.moniker=moniker
        self.wlearner = []
        self.alpha = np.zeros(self.T)
        self.model=VW(moniker=moniker,passes=10,loss='quadratic', csoaa=10)
    
    
    '''MNIST_DATA is a list of strings'''
    def fit(self, MNIST_DATA,Y):
        k = np.unique(Y)
        print len(k)
        print np.shape(MNIST_DATA)
        m = np.size(MNIST_DATA)
        '''In our case, the k is 10 for MNIST data set'''
        f = np.zeros((m, len(k)))
        C = np.zeros((m, len(k)))

        for t in range(self.T):
     
            '''choose cost matrix C'''
            # set values where l != yi

            C = np.exp(f - np.choose(Y, f.T)[:, np.newaxis])
            # set values where l == yi
            C[np.array(range(m)), Y] = 0
            d_sum = -np.sum(C, axis = 1)
            C[np.array(range(m)), Y] = d_sum

            #vw_cost is the cost matrix in vowpal wabbit conpatibel version
            csooa_data=self.transform(C,MNIST_DATA)
            print csooa_data[2]
            #csoaa should be a list of Instance rather than a list of strings
            #call vowpal wabbit for training a weak classifier.
            #self.wlearner[t] = VW("--coass 10 vw_cost -f csoaa.mm.model")

            self.wlearner[t]=self.train(csooa_data)
            
            #predicion on train set
            #htx is an array of prediction across the whole data
            _,htx=self.wlearner[t].predicion(csooa_data)
            #theta = weaklearner parameters
            #htx - predicted y and it is an array of predicted values
            
            #calculate delta using the predicions, cost matrix and f
            delta = -np.sum(C[np.array(range(m)), np.array(htx)])/np.sum(d_sum)
            #calculate alpha
            self.alpha[t] = 0.5 * np.log(1.0 * (1 + delta) / (1 - delta))
            #update f matrix
            f = f + alpha * (htx == Y)
    #output final classifier weights.
    
    
    
    '''vw_mnist is a list type and COST_MATRIX is a ndarray type'''
    def transform(self, COST_MATRIX, vw_mnist):
        n_samples, n_features = np.shape(COST_MATRIX)
        result = []
        for i in range(n_samples):
            tuple_exampe=vw_mnist[i].split('| ')
            feature_value=tuple_exampe[1]
            vw_csoaa_example=' '.join([' '.join([str(j)+':'+`COST_MATRIX[i,j]` for j in range(n_features) if COST_MATRIX[i,j] != 0]),'|',feature_value])
            Instance(vw_csoaa_example)
            result.append(Instance(vw_csoaa_example))

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
        with self.model.predicting():
            seen = 0
            for instance in instance_stream:
                self.model.push_instance(instance)
                instances.append(instance)
                seen += 1
        
        print '%s: predicted for %d data points' % (self.moniker, seen)
        predictions = list(self.model.read_predictions_())
        if seen != len(predictions):
            raise Exception("Number of labels and predictions do not match!  (%d vs %d)" % \
                            (seen, len(predictions)))
        return itertools.izip(instances, predictions)
    
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



if __name__ == '__main__':


    '''    for (instance, prediction) in SimpleModel('example1').train(instances).predict(instances):
        print prediction, instance
    '''
    path='/home/liguifan/Desktop/new_fold/data/vw_multiclass.valid'
    adaboost=adaboostMM('liguifan',path, )
    MNIST, Y=adaboost.read_MnistFile(path)
    print Y

    adaboost.fit(MNIST,Y)
    '''MNIST_DATA=read_MnistFile(path)
    fit(MNIST_DATA)'''