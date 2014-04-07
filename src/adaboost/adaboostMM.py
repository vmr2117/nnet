'''
adaboost.MM implementation.
'''

import numpy as np
from vowpal_porpoise import VW

class adaboostMM:
    def __init__(self, rounds = 5):
        self.T = rounds
        self.wlearner = []
        self.alpha = np.zeros(self.T)

    def fit(self, X,Y):
        k = np.unique(Y)
        m, _ = X.shape
        '''In our case, the k is 10 for MNIST data set'''
        f = np.zeros((m, k))
        C = np.zeros((m, k))
        for t in range(T):
            '''choose cost matrix C'''
            # set values where l != yi
            C = np.exp(f - np.choose(Y, f.T)[:, np.newaxis])
            # set values where l == yi
            C[np.array(range(m)), Y] = 0
            d_sum = -np.sum(C, axis = 1)
            C[np.array(range(m)), Y] = d_sum
            #vw_cost is the cost matrix in vowpal wabbit conpatibel version
            convert_cost_vw(C)
            #call vowpal wabbit for training a weak classifier.
            #self.wlearner[t] = VW("--coass 10 vw_cost -f csoaa.mm.model")
            a = VW("--coass 10 vw_cost -f csoaa.mm.model")
            #predicion on train set
            #htx = vw(X,Y,self.wlearner[t])
            htx=a.learn("-t -i csoaa.mm.model vw_cost")
            #theta = weaklearner parameters
            #htx - predicted y

            #calculate delta using the predicions, cost matrix and f
            delta = -np.sum(C[np.array(range(m)), htx])/np.sum(d_sum)
            #calculate alpha
            self.alpha[t] = 0.5 * np.log(1.0 * (1 + delta) / (1 - delta))
            #update f matrix
            f = f + alpha * (htx == Y)
        #output final classifier weights. 



    '''Write cost matrix to file vw_cost as the format vw wanted'''
    def convert_cost_vw(self, cost_matrix):
        f=open('vw_cost','w')
        n_samples,n_features=cost_matrix.shape
        for i in range(n_samples):
            #m_temp=[]
            for j in range(n_features):
                if cost_matrix[i][j]!=0:
                #m_temp.append(`j+1`+':'+`cost_matrix[i][j]`)
                    f.write(`j+1`+':'+`cost_matrix[i][j]`+' ')
            f.write('\n')
        f.close()

    def transform(self, X):
        n_samples, n_features = np.shape(X)
        result = []
        for i in range(n_samples):
            result.append({
                          str(j): X[i, j]
                          for j in range(n_features)
                          if X[i, j] != 0
                          })
        return result

    def train(self, instance_stream):
	model=VW(moniker='test',passes=10,loss='quadratic',csoaa=10)

	for instance in instance_stream:
		model.push_instance(instance)
		seen+=1
		if seen % 1000 ==0:
			print 'setreamed %d instances already' % seen
	print 'streaming finished'
	return self






print a
