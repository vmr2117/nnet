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
    def __init__(self,  moniker, path, passes,rounds = 5):
        self.T = rounds
        self.moniker=moniker
        self.wlearner = []
        self.alpha = np.zeros(rounds)
        self.model=VW(moniker=moniker, name= 'cache_d', passes=passes , csoaa=10)
    
    
    '''MNIST_DATA is a list of strings'''
    def fit(self, MNIST_DATA,Y):
        k = np.unique(Y)
        print k

        m = np.size(MNIST_DATA)
   
        '''In our case, the k is 10 for MNIST data set'''
        f = np.zeros((m, len(k)+1))
        C = np.zeros((m, len(k)+1))
        #vw_cost is the cost matrix in vowpal wabbit conpatibel version
        


        for t in range(self.T):
            '''choose cost matrix C'''
            # set values where l != yi
            C = np.exp(f - np.choose(Y, f.T)[:, np.newaxis])
            #(10000,1)

            # set values where l == yi
            C[np.array(range(m)), Y] = 0
            d_sum = -np.sum(C, axis = 1)
            C[np.array(range(m)), Y] = d_sum
            print np.shape(d_sum)
            #(10000,)
            
            #for x in csoaa_data:
            #     tempfile.write(str(x))
            # break

            #csoaa is a list of strings with the format vw takes 
            csoaa_data=self.transform(C,MNIST_DATA)

            for i in range(100):
                print 'csoaa format is ', csoaa_data[i]

            print 'current t is ', t
            
            #call vowpal wabbit for training a weak classifier.
            self.wlearner.append(self.train(csoaa_data))
            #_, prediction_file = tempfile.mkstemp(dir='.', prefix=self.model.get_prediction_file())
            temp_htx = self.wlearner[t].predict(csoaa_data)
            #htx is an array of prediction across the whole data in integer format
            htx=[int(i) for i in temp_htx]
            
            #calculate delta using the predicions, cost matrix and f
            delta = -np.sum(C[np.array(range(m)), np.array(htx)])/(-np.sum(d_sum))
            
            #calculate alpha
            self.alpha[t] = 0.5 * np.log(1.0 * (1 + delta) / (1 - delta))
            
            #update f matrix
            #for l in range(1,11):
            #        f[np.array(range(m)),l] = f[np.array(range(m)),l] + self.alpha[t] * (htx == l*np.ones(m))
            
            ind_vec_htx = np.zeros_like(f) 
            ind_vec_htx[np.array(range(m)), np.array(htx)] = self.alpha[t]
            print 'ALPHA', self.alpha[t]
            f += ind_vec_htx
            print 'dims: ',f.shape, ind_vec_htx.shape
            print 'current round data', float(sum(htx==Y))/m
    
    
    
    '''vw_mnist is a list type and COST_MATRIX is a ndarray type'''
    def transform(self, COST_MATRIX, vw_mnist):
        n_samples, n_features = np.shape(COST_MATRIX)
        result = []
        for i in range(n_samples):
            tuple_exampe=vw_mnist[i].split('| ')
            feature_value=tuple_exampe[1]
            vw_csoaa_example=' '.join([' '.join([str(j)+':'+`COST_MATRIX[i,j]` for j in range(1,n_features) if COST_MATRIX[i,j] != 0]),'|',feature_value])
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
        #print '%s: predicting' % self.moniker
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
        if seen != len(predictions):
           raise Exception("Number of labels and predictions do not match!  (%d vs %d)" % \
                            (seen, len(predictions)))
        return  predictions[:len(predictions)]
    

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
            class_set[m]= ord(ex[0])-48 +1
            m+=1
        examples.close()
        return (mnist_after,class_set)


    '''For this case, we have 10 classes <1...10>'''
    def ada_classifier(self, example):
        result=[self.F_T(example,i) for i in range(1,11)]
        print 'before choos the argmax', result
        return np.argmax(result)+1

    '''Output weighted combination of weak classifier F_T'''
    def F_T(self, example, class_ass):
        result=0
        for t in range(self.T):
            naive_result=self.wlearner[t].single_predict(example)
            result+=self.alpha[t]*(int(naive_result[0])==class_ass)
            print 'result is ', result, int(naive_result[0])
        return result


    def test_adaboost(self, file_path):
        y_est=[]
        examples=open(file_path,"r")
        for example in examples:
            y_est.append(self.ada_classifier(example))
            print 'class as ',self.ada_classifier(example)
        #accuracy_rate=float(sum(y_est==list(label)))/len(Y)
        return y_est

    def test(self, file_path):
        examples=open(file_path,"r")
        print self.wlearner[0].predict(examples)
        return self

    def test_naive(self, file_path):
        examples=open(file_path,"r")
        for example in examples:
            print self.wlearner[0].single_predict(example)
            print self.wlearner[1].single_predict(example)
        return self


if __name__ == '__main__':
    '''The location of the file we need to process'''
    current_directory=os.getcwd()
    filename='vw_multiclass.train'
    #filename='validation_part_original'
    path=os.path.join(current_directory, filename)

    test_file_name='rightclassNo'
    test_path=os.path.join(current_directory, test_file_name)

    T=5
    adaboost=adaboostMM('ML',path, 2, T )
    MNIST, Y=adaboost.read_MnistFile(path)
    adaboost.fit(MNIST,Y)

 

    '''Test on a set with 23 examples'''
    filename='rightclassNo'
    path=os.path.join(current_directory, filename)

    examples=open(path,"r")
    c=[]
    for example in examples:
        c.append(adaboost.ada_classifier(example))
    print c
    
    '''print out the weight for weak classifier'''
    for t in range(T):
        print adaboost.alpha[t]

    


