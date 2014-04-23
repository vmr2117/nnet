"""This module implements output functions for units in the output layer,
associated cost functions and their derivatives with respect to
pre-output-function sum. 
"""

def linear(X):
    """Linear output function.

    Parameters
    ----------
    X : array_like, TODO
        Scalar inputs to the output units. These are the values obtained after
        weighting the activations from the last hidden layer (LHL) with the
        weights mapping LHL to the output layer.

    Return
    ------
    X : array_like, 
        Return input as is. 
    """
    return X

def logistic(X):
    """Logistic output function.

    Parameters
    ----------
    X : array_like, TODO
        Scalar inputs to the output units. These are the values obtained after
        weighting the activations from the last hidden layer (LHL) with the
        weights mapping LHL to the output layer.

    Return
    ------
    X : array_like, TODO 
        Logistic function evaluated at the elements of X.
    """
    in_range = [X >= -45 and X <= 45]
    ne_range = [X < -45]
    ps_range = [X >  45]
    X[in_range] = 1 / (1 + np.exp(-x))
    X[ne_range] = 0
    X[ps_range] = 1
    return X

def soft_max(X):
    """Soft-Max output function.

    Parameters
    ----------
    X : array_like, TODO
        Scalar inputs to the output units. These are the values obtained after
        weighting the activations from the last hidden layer (LHL) with the
        weights mapping LHL to the output layer.

    Return
    ------
    soft_max_X : array_like, TODO 
        Soft-Max function evaluated at the elements of X.
    """
    M = X.max(axis=1)
    X -= M # subtract the max out
    exp_X = np.exp(X)
    exp_X_sum = exp_X.sum(axis=1)
    soft_max_X = exp_X / np.tile(exp_X_sum, (1, exp_X.shape[1]))
    return soft_max_X

def nll_softmax_cost(Y, P, theta):
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
    cost = np.sum(np.multiply(Y, - np.log(FX)))
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
    derv = np.sum(np.multiply(P, Y), axis = 1) - 1.0
    return derv

