"""This module implements output functions for units in the output layer,
associated cost functions and their derivatives with respect to
pre-output-function sum. 
"""
import numpy as np

def softmax(X):
    """Soft-Max output function.

    Parameters
    ----------
    X : array_like, shape(n_samples, n_inputs)
        Scalar inputs to the output units. These are the values obtained after
        weighting the activations from the last hidden layer (LHL) with the
        weights mapping LHL to the output layer.

    Return
    ------
    soft_max_X : array_like, TODO 
        Soft-Max function evaluated at the elements of X.
    """
    M = X.max(axis=1)
    tX = X - M[:, np.newaxis] # subtract the max out
    exp_X = np.exp(tX)
    exp_X_sum = exp_X.sum(axis=1)
    soft_max_X = exp_X / exp_X_sum[:, np.newaxis]
    #print X.shape, exp_X.shape, exp_X_sum.shape
    return soft_max_X

def nll_softmax_cost(Y, P):
    """Negative log likelihood cost associate with softmax output units.

    Parameters
    ----------
    Y : array_like, shape(n_samples, n_classes)
        Indicator vector of Targets for samples.

    P : array_like, shape(n_samples, n_classes)
        Predicted posterior probability of the samples.

    Return
    ------
    cost : float
        Negative log likelihood cost.
    """
    cost = np.sum(np.multiply(Y, - np.log(P)))
    return cost

def nll_softmax_cost_derv(P, Y):
    """Derivative of the negative log likelihood cost function with respect to
    pre-softmax sum.

    Parameters
    ----------
    P : array_like, shape(n_samples, n_classes)
        Predicted posterior probability of the samples.

    Y : array_like, shape(n_sample, n_classes)
        Indicator vector of Targets for samples.

    Return
    ------
    derv : array_like, shape(n_samples)
        Derivative of the negative log likelihood cost function with respect to
        the pre-softmax sum.
    """
    derv = P - Y
    return derv

def get_output_func(func_name):
    """Returns output function, costs associated with the output functions and
    the derivative of the cost functions with respect to the input sums of the
    output functions.

    Parameters
    ----------
    func_name : str
        Name of the output function. Currently supports the following output
        functions - 'softmax'.

    Return
    ------
    output_func : function
        Output function.

    cost_func : function
        Cost function associated with the output function.

    cost_func_derv : function
        Derivative of the cost function with respect to the input sum of the
        output function.
    """
    assert func_name in ['softmax'], 'Unknown output function'
    output_func = None
    cost_func = None
    cost_func_derv = None

    if func_name == 'softmax':
        output_func = softmax
        cost_func = nll_softmax_cost
        cost_func_derv = nll_softmax_cost_derv

    return output_func, cost_func, cost_func_derv
