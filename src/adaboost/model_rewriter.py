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
    nnet_weights = [np.concatenate(estimator.intercept_[:, np.newaxis],
        estimator.coef_).T for estimator in adaboost_model.estimators_]
    hidden_layer_wts = np.concatenate(tuple(nnet_weights), axis=0)
    layer_weights = [hidden_layer_wts, adaboost_model.estimator_weights_]
    pickle.dump(layer_weights, open(nnet_weights_file, 'wb'))

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('adaboostSAMME_model_file', help='adaboostSAMME model \
            file')
    parser.add_argument('nnet_weights_file', help='file to write neural net \
    weights')
    args = parser.parse_args()
    write_weights(args.adaboostSAMME_model_file, args.nnet_weights_file)
