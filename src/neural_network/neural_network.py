'''
This module contains a 1 hidden layer neural network for classification. 
'''
import argparse
import copy
import cPickle as pickle
import numpy as np
import time

from activation_functions import get_actv_func
from cost_functions import logistic_cost
from sklearn.utils import shuffle

class network:
    def __init__(self, actv_func, log = True):
        self.actv, self.actv_der = get_actv_func(actv_func)
        self.log = log

    def __massage_data(self, X, Y):
        '''
        Adds a constant '1' feature to all the samples in X and converts the
        target Y into a matrix of indicator vectors.

        Return: X, Y - modified sample feature matrix and target indicator
                vector.
        '''
        n_classes = np.unique(Y).size
        new_Y = np.zeros((Y.size, n_classes), dtype=int)
        new_Y[np.array(range(Y.size)), Y] = 1
        X = np.concatenate((np.ones(X.shape[0])[:, np.newaxis], X), axis=1)
        return X, new_Y

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
        '''
        Feed forward the input X through the network with weights theta and
        return the output z and activations at each layer: input, hidden and
        output.

        Return: Z, activations - list of z values and activations
        '''
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
        '''
        Calculates errors at all hidden layers using the error at output layer,
        Z and current network weights theta.

        Returns: errors - errors on all hidden and output layers.
        '''
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
        '''
        Estimates the average partial gradients at all samples in (X, Y) using
        back propagation algorithm.

        Return: p_derivs - partial gradients of the cost function
        '''
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
 
    def __numerical_gradient(self, theta, X, Y):
        '''
        Computes numerical gradient of the logistic cost function with respect
        to the parameters in theta.

        Return: grad - numerically computed gradient.
        '''
        s = time.time()
        EPS = 10e-5
        grad = [np.empty_like(weights) for weights in theta]
        for layer in range(len(theta)): 
            for (x,y), value in np.ndenumerate(theta[layer]):
                layer_wts_cp = copy.deepcopy(theta) 
                layer_wts_cp[layer][x][y] = value + EPS
                cost_1 = logistic_cost(Y, self.__predict(X, layer_wts_cp))
                layer_wts_cp[layer][x][y] = value - EPS
                cost_2 = logistic_cost(Y, self.__predict(X, layer_wts_cp))
                grad[layer][x][y] = (cost_1 - cost_2) / (2 * EPS)
        return grad

    def __update_weights(self, p_derivs, learning_rate, theta):
        '''
        Updates the current weights using the given partial derivatives and the
        learning rate.
        '''
        for layer in range(len(theta)): 
            theta[layer] -=  learning_rate * p_derivs[layer]

    def __sgd(self, tr_X, tr_Y, vd_X, vd_Y, theta, batch_size = 32,
              max_epochs = 500, validation_freq = 50000, random = True): 
        '''
        Performs mini-batch stochastic gradient descent(SGD) on the dataset X,Y
        for a 'max_epochs' epochs.

        Return: cost_err - list of training cost and validation error
        '''
        n_samples, _ = tr_X.shape 
        batch_idx = [(i * batch_size, (i + 1) * batch_size - 1) 
                        for i in range(n_samples / batch_size)] 
        if batch_idx[-1][1] < n_samples - 1:
            batch_idx.append((batch_idx[-1][1]+1, n_samples - 1))
        
        cost_err = {}
        epoch = 1
        while epoch < max_epochs:
            # validation
            if (epoch * n_samples) % validation_freq == 0: 
               vd_err = self.__evaluate(vd_X, vd_Y, theta)
               tr_err = self.__evaluate(tr_X, tr_Y, theta)
               cost_err[epoch] = (tr_err, vd_err)
               print 'Updates:', epoch * n_samples, 'Validation Error:', \
                     vd_err, 'Training Error:', tr_err

            # training on a epoch
            tr_X, tr_Y = shuffle(tr_X, tr_Y)
            for batch in batch_idx:
                X = tr_X[batch[0]:batch[1]]
                Y = tr_Y[batch[0]:batch[1]]
                p_derivs = self.__gradient(X, Y, theta)
                self.__update_weights(p_derivs, 0.001, theta)
            epoch += 1

        print "Total Updates: ", epoch * n_samples
        return cost_err
        
    def __predict(self, X, theta):
        '''
        Predicts the activations obtained for all classes under the current
        model.

        Return: acvt - matrix of activations for each example under the weights
                theta.
        '''
        r, _ = X.shape
        n_classes, _ = theta[-1].shape
        _, actv = self.__feed_forward(X, theta)
        return actv[-1]

    def __evaluate(self, X, Y, theta):
        '''
        Evaluates the error of the network with weights theta by testing on
        samples (X, Y). X should already have the bias constant for the samples
        and Y should be a indicator vector.

        Return: err - the error of the network on the given data (X, Y)
        '''
        err = (np.sum(np.argmax(Y, axis = 1) != np.argmax(self.__predict(X,
               theta), axis = 1)) / (1.0 * X.shape[0]))
        return err
    

    def check_gradient(self, X, Y, hidden_units = 100):
        '''
        Checks gradients computed by back propagation.
        
        Return: True/False - True if the gradients computed by back_prop are
                within 0.001 of the numerically computed gradients.
        '''
        X, Y = self.__massage_data(X, Y)
        _, n_features = X.shape
        _, n_classes = Y.shape
        theta = self.__random_weights(n_features, n_classes, hidden_units)
        bprop_grad = self.__gradient(theta, X, Y)
        num_grad = self.__numerical_gradient(theta, X, Y) 
        diff = [np.amax(np.absolute(gradl - dervl)) for gradl, dervl in
                zip(num_grad, bprop_grad)]
        return max(diff) < 10e-8

    def evaluate(self, X, Y, theta):
        '''
        Evaluates the error of the network with weights theta by testing on
        samples (X, Y) 

        Return: err - the error of the network on the given data (X, Y)
        '''
        X, _ = self.__massage_data(X, Y)
        err = (np.sum(Y != np.argmax(self.__predict(X, theta), axis = 1)) 
               / (1.0 * X.shape[0]))
        return err

    def train(self, X, Y, X_vd, Y_vd, hidden_units = None, theta = None): 
        '''
        Trains the network using Stochastic Gradient Descent. Initialize the
        network with the weights theta, if provided, else uses the hidden units
        parameter and generates weights theta randomly. Training data is assumed
        to be randomly shuffled already. 

        Return: cost_err - list of training cost and validation error
                theta - final weights of the network
        '''
        ok = (hidden_units is not None or theta is not None)
        assert ok, 'hidden units / weights missing'

        X, Y = self.__massage_data(X, Y)
        X_vd, Y_vd = self.__massage_data(X_vd, Y_vd)
        _, n_features = X.shape
        _, n_classes = Y.shape
        if not theta:
            theta = self.__random_weights(n_features, n_classes, hidden_units)

        cost_err = self.__sgd(X, Y, X_vd, Y_vd, theta)
        return cost_err, theta

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description = 'Check backprop gradient \
            computation by comparing with numerically computed gradient ')
    parser.add_argument('data_file', help = 'data file containing feature and \
            labels')
    parser.add_argument('hidden_units', help = 'number of hidden units -should \
                         be a multiple of the number of classes in the data \
                         set', type = int)
    args = parser.parse_args()
    nnet = network('logistic')
    data = pickle.load(open(args.data_file)) 
    assert nnet.check_gradient(data['X'], data['Y'], args.hidden_units), \
                                                    'Incorrect gradient!'
    print 'Gradient check passed'

