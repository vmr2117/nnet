'''
This module provides activation functions and their derivatives.
'''
import numpy as np

def tanh(x):
    """Hyperbolic tangent function.

    Parameters
    ----------
    x : float
        Input value.

    Return
    ------
    val : float
        Value of hyperbolic tangent function evaluated at x.
    """
    val = np.tanh(x)
    return val

def tanh_derivative(x):
    """ Derivative of hyperbolic tangent function.

    Parameters
    ----------
    x : float
        Input value.

    Return
    ------
    val : float 
        Value of derivative of hyperbolic tangent function at x.
    """
    val = 1.0 - np.tanh(x) ** 2
    return val

def sigmoid(x):
    """Sigmoid function.

    Parameters
    ----------
    x : float
        Input value.

    Return
    ------
    val : float 
        Value of sigmoid function at x.
    """
    val = None
    # handle overflows.
    if x < -45 : val = 0 
    elif x > 45 : val = 1
    else: val = 1 / (1 + np.exp(-x)) 
    return val

def sigmoid_derivative(x):
    """ Derivative of sigmoid function.

    Parameters
    ----------
    x : float
        Input value.

    Return
    ------
    val : float 
        Value of derivative of sigmoid function at x.
    """
    sig = sigmoid(x)
    val = sig * (1 - sig) 
    return val

def get_actv_func(func_name):
    """Returns activation function and its derivative 

    Parameters
    ----------
    func_name : str
        Name of the activation function.

    Return 
    ------
    func : function
        Activation function.

    func_derv : function
        Derivative of the activation function.
    """
    assert func_name in ['sigmoid', 'tanh'], 'Unknown activation function'
    func = None
    func_derv = None
    if func_name == 'sigmoid':
        func = sigmoid
        func_derv = sigmoid_derivative
    elif func_name == 'tanh':
        func = tanh
        func_derv = tanh_derivative
    return func, func_derv
