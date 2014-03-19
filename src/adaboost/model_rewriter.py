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
    nnet_weights.append(adaboost_model.estimator_weights_)
    pickle.dump(nnet_weights, open(nnet_weights_file, 'wb'))

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('adaboostSAMME_model_file', help='file ')
    wts, coefs = get_weights('../../models/numpy_array_multiclass.model')
    np.set_printoptions(suppress=True,precision=7)
    print 'Estimator weights:'
    print '------------------'
    print wts
    print 'Coeffs:'
    print '------------------'
    for coef in coefs:
        dim_1, dim_2 = np.nonzero(coef[0])
        print coef[0][np.array(dim_1), np.array(dim_2)]
    print 'Intercepts:'
    print '------------------'
    for coef in coefs:
        dim = np.nonzero(coef[1])
        print coef[1][np.array(dim)]
