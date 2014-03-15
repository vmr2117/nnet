'''
adaboost.MM implementation.
'''

import numpy as np

class adaboostMM:
    def __init__(rounds = 5):
        self.T = rounds

    def fit(X,Y):
        k = np.unique(Y)
        m, _ = X.shape
        f = np.zeroes(m, k)
        C = np.zeroes(m, k)
        alpha = np.zeroes(self.T)
        for t in range(T):
            #choose cost matrix
            C = np.zeroes(m, k) 
            C[,]
            #call vowpal wabbit for training a weak classifier
            #use the weaklearner to determing predicions on the train set.
            #calculate delta using the predicions, cost matrix and f
            #update alpha
            alpha[i] = 0.5 * np.log(1.0 * (1 + delta) / (1 - delta))
            #update f 
        #output final classifier weights. 

