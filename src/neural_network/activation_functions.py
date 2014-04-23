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

def logistic(x):
    """Logistic function.

    Parameters
    ----------
    x : float
        Input value.

    Return
    ------
    val : float 
        Value of logistic function at x.
    """
    val = None
    # handle overflows.
    if x < -45 : val = 0 
    elif x > 45 : val = 1
    else: val = 1 / (1 + np.exp(-x)) 
    return val

def logistic_derivative(x):
    """ Derivative of logistic function.

    Parameters
    ----------
    x : float
        Input value.

    Return
    ------
    val : float 
        Value of derivative of logistic function at x.
    """
    s = logistic(x)
    val = s * (1 - s) 
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
    assert func_name in ['logistic', 'tanh'], 'Unknown activation function'
    func = None
    func_derv = None
    if func_name == 'logistic':
        func = logistic
        func_derv = logistic_derivative
    elif func_name == 'tanh':
        func = tanh
        func_derv = tanh_derivative
    return func, func_derv
