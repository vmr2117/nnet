'''
This module provides activation functions and their derivatives.
'''
import numpy as np

def tanh(x):
    return np.tanh(x)

def tanh_derivative(x):
    return 1.0 - x ** 2

def logistic(x):
    return 1 / (1 + exp(-x)) 

def logistic_derivative(x):
    return logistic(x) * (1 - logistic(x))

def get_actv_func(func):
    '''
    Returns a tuple containing activation function 'func' and its derivative
    function. Available activation function are logistic and tanh. 
    '''
    assert func in ['logistic', 'tanh'], 'Unknown activatino function'
    if func == 'logistic':
        return (logistic, logistic_derivative)
    elif func == 'tanh':
        return (tanh, tanh_derivative)
