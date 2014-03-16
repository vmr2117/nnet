'''
adaboost.MM implementation.
'''

import numpy as np

class adaboostMM:
    def __init__(self, rounds = 5):
        self.T = rounds
        self.wlearner = []
        self.alpha = np.zeroes(self.T)

    def fit(self, X,Y):
        k = np.unique(Y)
        m, _ = X.shape
        f = np.zeroes(m, k)
        C = np.zeroes(m, k)
        for t in range(T):
            '''choose cost matrix C'''
            # set values where l != yi
            C = np.exp(f - np.choose(Y, f.T)[:, np.newaxis])
            # set values where l == yi
            C[np.array(range(m)), Y] = 0
            d_sum = -np.sum(C, axis = 1)
            C[np.array(range(m)), Y] = d_sum

            #call vowpal wabbit for training a weak classifier.
            self.wlearner[t] = vw()
            #predicion on train set
            htx = vw(X,Y,self.wlearner[t])
            #theta = weaklearner parameters
            #htx - predicted y

            #calculate delta using the predicions, cost matrix and f
            delta = -np.sum(C[np.array(range(m)), Y])/np.sum(d_sum)
            #calculate alpha
            self.alpha[t] = 0.5 * np.log(1.0 * (1 + delta) / (1 - delta))
            #update f matrix
            f = f + alpha * (htx == Y)
        #output final classifier weights. 

