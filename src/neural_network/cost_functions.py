'''
This module provides loss functions.
'''
import numpy as np

def log(Y, FX):
    '''
    Returns log loss.
    Y is a matrix of true labels. The examples are along the first dimesion and
    the multiclass labels are along the second dimension. Similarly, FX has the
    predicted labels; the first dimension had the examples and the second
    dimension has the probability value for the labels. 
    '''
    return -1 * np.sum(np.multiply(Y, np.log(FX)))

def square(Y, FX):
    '''
    Returns Square loss.
    Y is a matrix of true labels. The examples are along the first dimesion and
    the multiclass labels are along the second dimension. Similarly, FX has the
    predicted labels; the first dimension had the examples and the second
    dimension has the probability value for the labels. 
    '''
    return np.square(Y - FX)

