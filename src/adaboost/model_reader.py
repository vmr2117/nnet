'''
model_reader provides methods to extract weights from the model.
'''

import cPickle as pickle
import numpy as np

def load_model(adaboost_model_file):
    return pickle.load(open(adaboost_model_file))

def get_weights(adaboost_model_file):
    adaboost_model = load_model(adaboost_model_file)
    estimator_weights = adaboost_model.estimator_weights_
    estimator_coefs = [(estimator.coef_, estimator.intercept_) for 
            estimator in adaboost_model.estimators_]
    return (estimator_weights, estimator_coefs)


if __name__ == '__main__':
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
