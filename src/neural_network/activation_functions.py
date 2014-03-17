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
    return logistic(x) * (1 - logistix(x))
