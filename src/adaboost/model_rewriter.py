'''
model_reader provides methods to extract weights from the adaboostSAMME model
and rewrite them in a format suitable for initializing neural networks. It
pickles a list that contains weights of the neural network layer by layer.
'''

import argparse
import cPickle as pickle
import numpy as np


def write_weights(adaboost_model_file, nnet_weights_file):
    adaboost_model = pickle.load(open(adaboost_model_file)) 
    wl_weights = [estimator.coef_ for estimator in adaboost_model.estimators_] 
    wl_intercepts = [estimator.intercept_ for estimator in adaboost_model.estimators_] 
    theta_1 = np.concatenate(tuple(wl_weights), axis=0)
    bias_1 = np.concatenate(tuple(wl_intercepts), axis=0)
    

    n_classes = adaboost_model.n_classes_
    hidden_units = len(adaboost_model.estimators_) * n_classes 
    col_inds = np.arange(hidden_units).reshape((n_classes,-1),
                order='F').reshape((-1,))
    row_inds =np.tile(np.arange(n_classes),(hidden_units/n_classes,
                1)).T.reshape(-1)
    theta_2 = np.zeros((n_classes, hidden_units))
    theta_2[row_inds, col_inds] = np.tile(adaboost_model.estimator_weights_,
            (n_classes,))

    bias_2 = np.zeros((n_classes,))

    theta = [theta_1, theta_2]
    bias = [bias_1, bias_2]
    pickle.dump([theta, bias], open(nnet_weights_file, 'wb'))

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='''Convert adaboostSAMME model 
            weights into neural network weights. All the parameters of the weak
            learners form the weights for the activation units in the hidden  
            layer and the weak learner alphas form the weights for the last 
            layer''')
    parser.add_argument('adaboostSAMME_model_file', help='adaboostSAMME model \
            file')
    parser.add_argument('nnet_weights_file', help='file to write neural net \
    weights')
    args = parser.parse_args()
    write_weights(args.adaboostSAMME_model_file, args.nnet_weights_file)
