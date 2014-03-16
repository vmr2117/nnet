'''
Module to construct a fully connected neural network.
'''
import numpy as np

class network:
    def __init__(self, layer_units):
        self.layer_units = layer_units 
        self.num_wt = 0
        for i in range(1, layer_units.size):
            num_wt += layer_units[i] * layer_units[i-1] + layer_units[i] + 1

    def initialize(self):
        '''
        Randomly initializes the weights of a neural network
        '''
        self.wt = np.random.rand(self.num_wt)

    def set_weights(self, weights):
        '''
        Initializes the network with the given weights if the network had not
        been already intialized.
        '''
        if weights.size != self.num_wt: return False 
        if hasattr(self, 'wt'): return False
        self.wt = weights 
        return True

    def train(stop_early = True, weight_decay = True, X, Y):


    def predict(X):
        pass
