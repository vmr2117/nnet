""" Feed Forward Neural Network for multiclass classification.
"""
import argparse
import copy
import cPickle as pickle
import numpy as np

from activation_functions import get_actv_func
from cost_functions import logistic_cost
from sklearn.utils import shuffle

class FFNeuralNetwork:
    """A Feed Forward Neural Network for multiclass classification using
    multiclass logistic cost function. The number of hidden layers, number of
    hidden units and whether to include the biases are inferred from the
    initial weights. The activation function to use for the network should be
    specified through the constructor.
    """
    def __init__(self, actv_func):
        """Initialize the Feed Forward Neural Network.

        Parameters
        ----------
        actv_func : string
            Name of the activation function to use - 'logistic' or 'tanh'. 
        """
        self.actv, self.actv_der = get_actv_func(actv_func)

    def __massage_data(self, X, Y):
        """Adds a constant unit feature to all the samples in X and converts
        the target vector Y into a matrix of indicator vectors.

        Parameters
        ----------
        X : array_like, shape(n_samples, n_features)
            Data.

        Y : array_like, shape(n_samples) 
            Targets for samples in X.

        Return
        ------
        X_m : array_like, shape(n_samples, n_features + 1)
            Data with a constant unit feature to accomodate bias.

        Y_m : array_like, shapee(n_samples, n_classes)
            Indicator vectors of targets for samples in X.
        """
        n_classes = np.unique(Y).size
        Y_m = np.zeros((Y.size, n_classes), dtype=int)
        Y_m[np.array(range(Y.size)), Y] = 1
        X_m = np.concatenate((np.ones(X.shape[0])[:, np.newaxis], X), axis=1)
        return X_m, Y_m

    def __random_weights(self, n_features, n_classes, hidden_units):
        '''
        Initializes weights for 1 hidden layer neural network architecture.
        Weights are chosed uniformly from [-0.5, 0.5).

        Return: theta - weight matrix for the neural network.
        ''' 
        theta = []
        theta.append(np.random.rand(hidden_units, n_features) -
                0.5) # theta 1 - maps input layer to hidden layer
        col_inds = np.arange(hidden_units).reshape((n_classes,-1),
                order='F').reshape((-1,)) 
        row_inds =np.tile(np.arange(n_classes),(hidden_units/n_classes,
                1)).T.reshape(-1) 
        weights = np.zeros((n_classes, hidden_units))
        weights[row_inds, col_inds] = np.random.rand(hidden_units) - 0.5
        theta.append(weights) # theta 2 - maps hidden layer to output layer
        return theta

    def __feed_forward(self, X, theta):
        """Feed forward the input X through the network with weights theta.

        Parameters
        ----------
        X : array_like, shape(n_samples, n_features)
            Training data.

        theta : list(array_like)
            List of weights that map inputs to outputs.

        Return
        ------
        Z : list(array_like)
            Activation function arguments at each layer.

        activations : list(array_like)
            Activations at each layer.
        """
        Z = [X.T]
        activations = [X.T]
        for l_wt in theta: 
            Z.append(np.dot(l_wt, activations[-1]))
            activations.append(self.actv(Z[-1]))

        for ind in range(len(Z)): 
            Z[ind] = Z[ind].T
            activations[ind] = activations[ind].T

        return (Z, activations) 

    def __back_propagate(self, Z, error, theta):
        """Calculates errors at all hidden layers using the error at output
        layer, Z and current network weights theta.

        Parameters
        ----------
        Z : list(array_like)
            Activation function arguments at each layer.

        error : array_like, shape(n_output_units)
            Error in the output layer.

        Return
        -------
        errors : list(array_like)
            Errors on all hidden and output layers.
        """
        n_layers = len(theta) + 1
        errors = [error]

        # compute errors on hidden layers from o/p to i/p direction
        for ind in range(n_layers - 2):
            layer_ind = -(1 + ind)
            wt = theta[layer_ind]
            z = Z[layer_ind - 1]
            next_layer_err = errors[-1]
            errors.append(np.multiply(np.dot(next_layer_err, wt),
                self.actv_der(z))) 
        return list(reversed(errors))

    def __gradient(self, X, Y, theta):
        """Estimates the averaged partial gradients of the cost function with
        respect to the parameters in theta using back propagation algorithm.

        Parameters
        ----------
        X : array_like, shape(n_samples, n_features)
            Data.

        Y : array_like, shape(n_samples, n_classes)
            Targets for samples in X.

        theta : list(array_like)
            Weights for all layers.

        Return
        ------
        p_derivs : list(array_like)
            Partial derivatives matrix for each layer.
        """
        Z, activations = self.__feed_forward(X, theta)
        deltas = self.__back_propagate(Z, activations[-1] - Y, theta)
        p_derivs = None
        if len(X.shape) == 2:
            n_samples, _= X.shape
            p_derivs = [(np.einsum('ij,ik->jk',deltas[layer], activations[layer]) 
                         / n_samples)
                        for layer in range(0, len(activations) - 1)]
        else:
            p_derivs = [np.outer(deltas[layer], activations[layer]) 
                                    for layer in range(0, len(activations) - 1)] 
        return p_derivs
 
    def __numerical_gradient(self, X, Y, theta, eps):
        """Computes numerical gradient of the logistic cost function with
        respect to the parameters in theta.

        Parameters
        ----------
        X : array_like, shape(n_samples, n_features)
            Data.

        Y : array_like, shape(n_samples, n_classes)
            Targest for the samples in X.

        theta : list(array_like)
            Weights for all layers.

        eps : float
            Small delta for numerical gradient computation.

        Return
        ------
        grad - numerically computed partial gradients.
        """
        grad = [np.empty_like(weights) for weights in theta]
        for layer in range(len(theta)): 
            for (x,y), value in np.ndenumerate(theta[layer]):
                layer_wts_cp = copy.deepcopy(theta) 
                layer_wts_cp[layer][x][y] = value + eps
                cost_1 = logistic_cost(Y, self.__output_actv(X, layer_wts_cp))
                layer_wts_cp[layer][x][y] = value - eps
                cost_2 = logistic_cost(Y, self.__output_actv(X, layer_wts_cp))
                grad[layer][x][y] = (cost_1 - cost_2) / (2 * eps)
        return grad

    def __update_weights(self, p_derivs, learning_rate, theta):
        """Updates the current weights using the given partial derivatives and
        the learning rate.

        Parameters
        ----------
        p_derivs : list(array_like)
            Partial derivates of the cost function with respect to parameters
            in theta.

        learning_rate : float
            Learning rate.

        theta : list(array_like)
            Weights for all the layers.
        """
        for layer in range(len(theta)): 
            theta[layer] -=  learning_rate * p_derivs[layer]

    def __output_actv(self, X, theta):
        """Returns the output layer activations for the samples in X using the
        layer weights theta.

        Parameters
        ----------
        X : array_like, shape(n_samples, n_features)
            Data.

        theta :
            Weights of all layers.

        Return
        ------
        acvt : array_like, shape(n_samples, n_classes)
            Output layer activation for the samples.
        """
        r, _ = X.shape
        n_classes, _ = theta[-1].shape
        _, actv = self.__feed_forward(X, theta)
        return actv[-1]

    def __evaluate(self, X, Y, theta):
        """Evaluates the network on the test samples in X and returns the
        classification error.

        Parameters
        ----------
        X : array_like, shape(n_samples, n_features)
            Data. X should be massaged already - should contain a unit feature
            to account for bias term.

        Y : array_like, shape(n_samples, n_classes)
            Indicator vectors of Targets for the samples in X.

        theta : list(array_like)
            Weights of all the layers.

        Return
        ------
        err : float
            Classification error for the samples in X.
        """
        err = (np.sum(np.argmax(Y, axis = 1) != np.argmax(self.__predict(X,
               theta), axis = 1)) / (1.0 * X.shape[0]))
        return err
    

    def test_backprop_gradients(self):
        """Checks gradients computed by back propagation.
        
        Return
        ------
        pass : True/False
            True if the gradients computed by back_prop are within 10e-8 of the
            numerically computed gradients.
        """
        # prepare dummy data.
        X = np.random.uniform(size = (1000,784)) 
        Y = np.random.randint(0, high = 10, size=(1000))
        wt = np.sqrt(6) / np.sqrt(20)
        weights_1 = np.random.uniform(-wt, high = wt, size = (10, 785)) 
        weights_2 = np.random.uniform(-wt, high = wt, size = (10, 10)) 
        weights_3 = np.random.uniform(-wt, high = wt, size = (10, 10)) 
        weights_4 = np.random.uniform(-wt, high = wt, size = (10, 10)) 
        theta = [weights_1, weights_2, weights_3, weights_4]

        X, Y = self.__massage_data(X, Y)
        bprop_grad = self.__gradient(X, Y, theta)
        num_grad = self.__numerical_gradient(X, Y, theta, 10e-5) 
        diff = [np.amax(np.absolute(gradl - dervl)) for gradl, dervl in
                zip(num_grad, bprop_grad)]
        return max(diff) < 10e-10

    def test(self, X, Y, theta):
        """Tests the performance of network with weights theta on the samples
        in X.

        Parameters
        ----------
        X : array_like, shape(n_samples, n_features)
            Testing data.

        Y : array_like, shape(n_samples)
            Target for the testing samples.

        Return
        ------
        err - error of the network on the given data (X, Y)
        """
        X, _ = self.__massage_data(X, Y)
        err = (np.sum(Y != np.argmax(self.__output_actv(X, theta), axis = 1)) 
               / (1.0 * X.shape[0]))
        return err

    def train(self, db_writer, tr_X, tr_Y, vd_X, vd_Y, theta, batch_size = 32,
            max_epochs = 300, vd_freq = 1563):
        """Trains the network using mini-batch Stochastic Gradient Descent.

        Parameters
        ----------
        db_writer : DBWriter
            Object used to write parameters measured during training.

        tr_X : array_like, shape (n_samples, n_features) 
            Training data.

        tr_Y : array_like, shape (n_samples)
            Targets for the training samples.

        vd_X : array_like, shape (n_samples, n_features)
            Validation data.

        vd_Y : array_like, shape (n_samples)
            Targets for the validation samples.

        theta : list(array_like)
            Initial weights for the Network. Each array_like elements maps
            output from previous layer to the inputs of the next layer. The
            layer weights are arranged from left to right.

        batch_size : int
            Number of samples to use for each weight update step. The default
            value is 32.

        max_epochs : int
            Number of epochs to train the network. The default value is 300.
                     
        vd_freq : int
            Frequency of validation in units of number of weight updates. The
            default value is 1563.

        Return
        ------
        theta : list(array_like)
            Weights that gave the least validation error through the training
            phase.
        """
        tr_X, tr_Y = self.__massage_data(tr_X, tr_Y)
        vd_X, vd_Y = self.__massage_data(vd_X, vd_Y)
        # generate batch idx for training
        n_samples, _ = tr_X.shape 
        batch_idx = [(i * batch_size, (i + 1) * batch_size - 1) 
                        for i in range(n_samples / batch_size)] 
        if batch_idx[-1][1] < n_samples - 1:
            batch_idx.append((batch_idx[-1][1]+1, n_samples - 1))
        
        best_vd_err = np.inf
        best_theta = None
        epoch = 0
        while epoch < max_epochs:
            tr_X, tr_Y = shuffle(tr_X, tr_Y)
            for num, batch in enumerate(batch_idx):
                # check training and validation error based on the requested
                # frequency
                batch_iters = epoch * len(batch_idx) + num
                if batch_iters % vd_freq == 0:
                    tr_err = self.__evaluate(tr_X, tr_Y, theta)
                    vd_err = self.__evaluate(vd_X, vd_Y, theta)
                    db_writer.write(batch_iters, tr_err, vd_err)
                    if vd_err < best_vd_err:
                        best_vd_err = vd_err
                        best_theta = theta
                # update weights
                X = tr_X[batch[0]:batch[1]]
                Y = tr_Y[batch[0]:batch[1]]
                p_derivs = self.__gradient(X, Y, theta)
                self.__update_weights(p_derivs, 0.02, theta)
            epoch += 1
        return best_theta

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description = 'Check backprop gradient \
            computation by comparing with numerically computed gradient ')
    args = parser.parse_args()
    nnet = FFNeuralNetwork('sigmoid')
    assert nnet.test_backprop_gradients(), 'Incorrect gradient!'
    print 'Gradient check passed'

