'''
This module provides loss functions.
'''
import numpy as np

def logistic_cost(Y, FX, theta = None, reg_lambda = 0):
    '''
    Returns log loss.
    Y is a matrix of true labels. The examples are along the first dimesion and
    the multiclass labels are along the second dimension. Similarly, FX has the
    predicted labels; the first dimension had the examples and the second
    dimension has the probability value for the labels. theta is 1 dimensional
    vector of parameters to be l2 regularized
    '''
    cost = - np.sum(np.multiply(Y, np.log(FX)) + np.multiply(1 - Y, np.log(1
                       - FX))) / Y.shape[0]
    reg_cost = 0
    if reg_lambda and theta:
        reg_cost = np.sum(np.square(theta)) * reg_lambda * 1.0 / (2 * m)
    return cost + reg_cost

