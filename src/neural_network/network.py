'''
Module to construct a fully connected neural network.
'''
import numpy as np

class network:
    def __init__(self, layer_units):
        self.layer_units = layer_units
        self.layer_weights = None

    def __random_init():
        '''
        Randomly initialize the weights of neural network.
        '''
        pass

    def __check_weights(layer_weights):
        allok = True:
        allok = len(layer_units) -1 == len(layer_weights)
        # check every layer for consistent weights
        return allok

    def __weights_init(self, layer_weights):
        '''
        Initialize the weights of the neural network with the given weights.
        Validata num_classes, num_units and weights with the architecture
        specified through layer_units.
        '''
        assert self.__check(layer_weights), 'weights incompatible with the\
            network acchitecture'
        self.layer_weights = layer_weights

    def train(self, X, Y, layer_weights = None):
        if layer_weights == None:self.__random__init()
        else self.__weights__init()

    def predict(self, X):
        pass
