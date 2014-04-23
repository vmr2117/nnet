"""This module implements various output function for the final layer in Neural
Networks
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
