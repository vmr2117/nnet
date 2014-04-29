""" Feed Forward Neural Network for multiclass classification.
"""
import argparse
import copy
import cPickle as pickle
import numpy as np
import time

from activation_functions import get_actv_func
from output_functions import get_output_func
from sklearn.utils import shuffle
from data_structures import Perf, Distribution

class FFNeuralNetwork:
    """A Feed Forward Neural Network.

    Usage
    -----
    Train:

    ff_nnet = FFNeuralNetwork()
    ff_net.set_activation_func('tanh')
    ff_net.set_output_func('softmax')
    ff_net.initialize(theta, bias)
    btheta,bbias = ff_net.train(db_writer, tr_X, tr_Y, vd_X, vd_Y, 32, 200,
                                1563)

    Test:
    ff_nnet = FFNeuralNetwork()
    ff_net.set_activation_func('tanh')
    ff_net.set_output_func('softmax')
    ff_net.initialize(btheta, bbias)
    print ff_net.test(ts_X, ts_Y)

    """
    def __init__(self):
        self.actv_func = None
        self.actv_func_derv = None
        self.output_func = None
        self.output_func_derv = None
        self.theta = None
        self.bias = None

        self.best_vd_err = np.inf
        self.best_theta = None
        self.best_bias = None
        self.perf_writer = None
        self.debug_writer = None

    def set_activation_func(self, actv_func):
        """ Sets the activation function for all hidden units in all hidden
        layers.

        Parameters
        ----------
        actv_func : string
            Name of the activation function to use - 'logistic' or 'tanh'. 
        """
        self.actv, self.actv_derv = get_actv_func(actv_func)

    def set_output_func(self, output_func):
        """Sets output function for all output units.

        Parameters
        ----------
        output_func : string
            Name of the output function to use - 'softmax'. 
        """
        self.output, self.cost, self.cost_derv = get_output_func(output_func)

    def set_perf_writer(self, perf_writer):
        """Sets the writer object which is used by the network to dump
        performance while its training.

        Parameters
        ----------
        perf_writer: object, type(PerfWriter)
        """
        self.perf_writer = perf_writer

    def set_debug_writer(self, debug_writer):
        """Sets the writer object which is used to dump various debug
        parameters after each evaluation cycle. 
        
        Parameters
        ----------
        debug_writer: object type(DebugWriter)
        """
        self.debug_writer = debug_writer

    def initialize(self, theta, bias):
        """ Initializes the weights of the network.

        theta : list(array_like), shape(n_units, n_inputs)
            List of weights mapping consecutive layers ordered from input to
            output layers.

        bias : list(array_like), shape(n_units)
            List of biases for hidden layers and output layers from ordered
            from input to output layers
        """
        self.theta = theta
        self.bias = bias

    def __create_indicator_vectors(self, Y):
        """Converts the target vector Y into a matrix of indicator vectors.

        Parameters
        ----------
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
        return Y_m

    def __feed_forward(self, X):
        """Feed forward the input X through the network with weights theta.

        Parameters
        ----------
        X : array_like, shape(n_samples, n_features)
            Training data.

        Return
        ------
        Z : list(array_like)
            Pre-activation sums at each layer. The first entry is the input
            feature itself.

        activations : list(array_like)
            Activations at each layer. The first entry is the input feature
            itself.
        """
        n_layers = len(self.theta)
        Z = [X.T]
        activations = [X.T]
        for layer in (range(n_layers - 1)): 
            Z.append(np.dot(self.theta[layer], activations[-1]) 
                     + self.bias[layer][:, np.newaxis])
            activations.append(self.actv(Z[-1]))

        for ind in range(len(Z)): 
            Z[ind] = Z[ind].T
            activations[ind] = activations[ind].T

        # Apply output function.
        pre_output_sum = (np.dot(self.theta[-1], activations[-1].T) 
                          + self.bias[-1][:, np.newaxis]).T
        Z.append(pre_output_sum)
        activations.append(self.output(pre_output_sum))

        return (Z, activations) 

    def __back_propagate(self, Z, output_error):
        """Calculates errors at all hidden layers using the error at output
        layer, Z and current network weights theta.

        Parameters
        ----------
        Z : list(array_like), shape(n_units)
            Pre-activation sums at all layers.

        error : array_like, shape(n_output_units)
            Error in the output layer. dC/dzL

        Return
        -------
        errors : list(array_like), shape(n_units)
            Errors on all hidden and output layers.
        """
        errors = [output_error]
        # compute errors on hidden layers from o/p to i/p direction
        for layer_ind in range(len(Z) - 2, 0, -1):
            next_layer_err = errors[-1]
            curr_wt = self.theta[layer_ind]
            curr_z = Z[layer_ind]
            errors.append(np.multiply(np.dot(next_layer_err, curr_wt),
                                      self.actv_derv(curr_z))) 
        return list(reversed(errors))

    def __gradient(self, X, Y):
        """Estimates the averaged partial gradients of the cost function with
        respect to the parameters in theta using back propagation algorithm.

        Parameters
        ----------
        X : array_like, shape(n_samples, n_features)
            Data.

        Y : array_like, shape(n_samples, n_classes)
            Targets for samples in X.

        Return
        ------
        theta_grad : list(array_like), shape(n_units, n_inputs)
            Partial gradients of the cost function with respect to theta.

        bias_grad : list(array_like), shape(n_units)
            Partial gradients of the cost function with respect to bias. 
        """
        Z, activations = self.__feed_forward(X)
        deltas = self.__back_propagate(Z, self.cost_derv(activations[-1], Y))
        theta_grad = None
        bias_grad = None
        if len(X.shape) == 2:
            theta_grad = [(np.einsum('ij,ik->jk',deltas[layer],
                                       activations[layer]))
                            for layer in range(len(self.theta))]
            bias_grad = [deltas[layer].sum(axis = 0) 
                            for layer in range(len(self.bias))]
        else:
            theta_grad = [np.outer(deltas[layer], activations[layer]) 
                            for layer in range(len(self.theta))] 
            bias_grad = deltas

        return theta_grad, bias_grad

    def __update_weights(self, theta_grad, bias_grad, learning_rate,
            learn_only_last = False):
        """Updates the current weights using the given partial derivatives and
        the learning rate.

        Parameters
        ----------
        theta_grad : list(array_like), shape(n_units, n_inputs)
            Partial gradients of the cost function with respect to theta.

        bias_grad : list(array_like), shape(n_units)
            Partial gradients of the cost function with respect to bias.

        learning_rate : float
            Learning rate.

        learn_only_last : boolean
            If true, returns only the gradients of weights mapping to the
            output layer and keep other gradients zero. Default is False.
        """
        st = 0
        if learn_only_last:
            st = len(self.theta) - 1

        for layer in range(st, len(self.theta)): 
            self.theta[layer] -=  learning_rate * theta_grad[layer]
            self.bias[layer] -= learning_rate * bias_grad[layer]

    def __evaluate(self, X, Y):
        """Evaluates the network on the test samples in X and returns the
        classification error.

        Parameters
        ----------
        X : array_like, shape(n_samples, n_features)
            Data.

        Y : array_like, shape(n_samples, n_classes)
            Indicator vectors of Targets for the samples in X.

        Return
        ------
        err : float
            Classification error for the samples in X.
        """
        err = (np.sum(np.argmax(Y, axis = 1) 
                      != np.argmax(self.__feed_forward(X)[1][-1],
                                   axis = 1)) 
               / (1.0 * X.shape[0]))
        return err
    
    def __monitor(self, iter_no, tr_X, tr_Y, vd_X, vd_Y):
        """Evaluates the network for training and validation error.
        
        Parameters
        ----------
        iter_no : int
            Iteration number.

        tr_X : array_like, shape(n_samples, n_feature)
            Training data.

        tr_Y : array_like, shape(n_samples, n_classes)
            Targets for samples in tr_X. 

        vd_X : array_like, shape(n_samples, n_feature)
            Validation data.

        vd_Y : array_like, shape(n_samples, n_classes)
            Targets for samples in vd_X. 

        Return
        ------
        tr_err : float
            Training error.
        """
        # validation 
        tr_err = self.__evaluate(tr_X, tr_Y)
        vd_err = self.__evaluate(vd_X, vd_Y)
        if self.perf_writer:
            self.perf_writer.write(Perf(iter_no, tr_err, vd_err))

        if vd_err < self.best_vd_err:
            # update best parameters
            self.best_vd_err = vd_err
            for ind in range(len(self.theta)):
                self.best_theta[:][ind] = self.theta[ind]
                self.best_bias[:][ind] = self.bias[ind]

        # activation distribution
        if self.debug_writer: 
            num = int((3.0/100) * vd_X.shape[0])
            sm_vd_X = vd_X[:num]
            _, activations = self.__feed_forward(sm_vd_X) 
            for i, actv in enumerate(activations[1:]):
                self.debug_writer.write(Distribution('actv', iter_no, i+1,
                                        np.mean(actv), np.std(actv)))

        return tr_err

    def test(self, X, Y):
        """Tests the performance of network.

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
        Y = self.__create_indicator_vectors(Y)
        return self.__evaluate(X, Y)

    def train(self, tr_X, tr_Y, vd_X, vd_Y, batch_size= 32,
              max_epochs = 300, vd_freq = 1563, learn_only_last = False):
        """Trains the network using mini-batch Stochastic Gradient Descent.

        Parameters
        ----------
        tr_X : array_like, shape (n_samples, n_features) 
            Training data.

        tr_Y : array_like, shape (n_samples)
            Targets for the training samples.

        vd_X : array_like, shape (n_samples, n_features)
            Validation data.

        vd_Y : array_like, shape (n_samples)
            Targets for the validation samples.

        batch_size : int
            Number of samples to use for each weight update step. The default
            value is 32.

        max_epochs : int
            Number of epochs to train the network. The default value is 300.
                     
        vd_freq : int
            Frequency of validation in units of number of weight updates. The
            default value is 1563.

        learn_only_last : boolean
            If true, returns only the gradients of weights mapping to the
            output layer and keep other gradients zero. Default is False.

        Return
        ------
        best_theta : list(array_like)
            Best theta found.

        best_bias : list(array_like)
            Best bias found.
        """
        tr_Y = self.__create_indicator_vectors(tr_Y)
        vd_Y = self.__create_indicator_vectors(vd_Y)
        # generate batch idx for training
        n_samples, _ = tr_X.shape 
        batch_idx = [(i * batch_size, (i + 1) * batch_size - 1) 
                        for i in range(n_samples / batch_size)] 
        if batch_idx[-1][1] < n_samples - 1:
            batch_idx.append((batch_idx[-1][1]+1, n_samples - 1))
        
        # initialization 
        self.best_theta = [np.empty_like(theta) for theta in self.theta]
        self.best_bias = [np.empty_like(bias) for bias in self.bias]
        best_vd_err = np.inf
        tr_err = np.inf
        epoch = 0

        # training - minibatch SGD
        while epoch < max_epochs and tr_err > 0:
            tr_X, tr_Y = shuffle(tr_X, tr_Y)
            for num, batch in enumerate(batch_idx):
                # monitor network performance
                b_iters = epoch * len(batch_idx) + num
                if b_iters % vd_freq == 0:
                    tr_err = self.__monitor(b_iters, tr_X, tr_Y, vd_X, vd_Y)
                # update weights
                X = tr_X[batch[0]:batch[1]]
                Y = tr_Y[batch[0]:batch[1]]
                theta_derivs, bias_derivs = self.__gradient(X, Y)
                self.__update_weights(theta_derivs, bias_derivs, 0.001,
                                      learn_only_last)
            epoch += 1
        return self.best_theta, self.best_bias

    def __numerical_gradient(self, X, Y, eps):
        """Computes numerical gradient of the cost function with respect to
        theta and bias.

        Parameters
        ----------
        X : array_like, shape(n_samples, n_features)
            Data.

        Y : array_like, shape(n_samples, n_classes)
            Targest for the samples in X.

        eps : float
            Small delta for numerical gradient computation.

        Return
        ------
        theta_grad : list(array_like), shape(n_units, n_inputs)
            List of partial gradients of cost function with respect to weights
            in theta.

        bias_grad : list(array_like), shape(n_units)
            List of partial gradients of cost function with respect to weights
            in bias.
        """
        theta_grad = [np.empty_like(weights) for weights in self.theta]
        for layer in range(len(self.theta)): 
            for (x,y), value in np.ndenumerate(self.theta[layer]):
                self.theta[layer][x][y] = value + eps
                cost_1 = self.cost(Y, self.__feed_forward(X)[1][-1])
                self.theta[layer][x][y] = value - eps
                cost_2 = self.cost(Y, self.__feed_forward(X)[1][-1])
                theta_grad[layer][x][y] = (cost_1 - cost_2) / (2 * eps)
                self.theta[layer][x][y] = value

        bias_grad = [np.empty_like(weights) for weights in self.bias]
        for layer in range(len(self.bias)): 
            for (x), value in np.ndenumerate(self.bias[layer]):
                self.bias[layer][x] = value + eps
                cost_1 = self.cost(Y, self.__feed_forward(X)[1][-1])
                self.bias[layer][x] = value - eps
                cost_2 = self.cost(Y, self.__feed_forward(X)[1][-1])
                bias_grad[layer][x] = (cost_1 - cost_2) / (2 * eps)
                self.bias[layer][x] = value
        return theta_grad, bias_grad

    def test_backprop(self):
        """Checks gradients computed by back propagation.
        
        Return
        ------
        test_pass : True/False
            True if the gradients computed by back_prop are within 10e-6 of the
            numerically computed gradients.

        max_diff : float
            Maximum difference found in all of the computed gradients.

        """
        # prepare dummy data.
        X = np.random.uniform(size = (1000,784)) 
        Y = np.random.randint(0, high = 10, size=(1000))
        wt = np.sqrt(6) / np.sqrt(20)
        weights_1 = np.random.uniform(-wt, high = wt, size = (10, 784)) 
        weights_rest =[ np.random.uniform(-wt, high = wt, size = (10, 10)) 
                        for _ in range(3) ]
        theta = [weights_1] + weights_rest
        bias = [np.random.uniform(-wt, high = wt, size = (10))
                        for _ in range(4)]
        self.theta = theta
        self.bias = bias
        Y = self.__create_indicator_vectors(Y)
        bp_th_grad, bp_bias_grad = self.__gradient(X, Y)
        num_th_grad, num_bias_grad = self.__numerical_gradient(X, Y, 10e-5) 
        diff_1 = [np.amax(np.absolute(gradl - dervl)) 
                for gradl,dervl in zip(num_th_grad, bp_th_grad)]
        diff_2 = [np.amax(np.absolute(gradl - dervl)) 
                for gradl,dervl in zip(num_bias_grad, bp_bias_grad)]
        max_diff = max(max(diff_1), max(diff_2))
        test_pass = max_diff < 10e-6
        return test_pass, max_diff

if __name__ == '__main__':
    ff_nnet = FFNeuralNetwork()
    ff_nnet.set_activation_func('tanh') 
    ff_nnet.set_output_func('softmax')
    passed, max_diff = ff_nnet.test_backprop()
    print 'tanh activations'
    print 'Max difference :', max_diff
    assert passed, 'Incorrect gradient!'
    print 'Gradient check passed'

    ff_nnet.set_activation_func('logistic') 
    ff_nnet.set_output_func('softmax')
    passed, max_diff = ff_nnet.test_backprop()
    print 'logistic activations'
    print 'Max difference :', max_diff
    assert passed, 'Incorrect gradient!'
    print 'Gradient check passed'

