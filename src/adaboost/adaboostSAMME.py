'''
adaboost.py script can be used to train and test adaboost with a SGDclassifier
for mnist data. Look up commandline options for more information.
'''
import argparse
import cPickle as pickle
import numpy as np
    
from sklearn.ensemble import AdaBoostClassifier
from sklearn.linear_model import SGDClassifier
from sklearn.metrics import accuracy_score

def write(model, model_file):
    '''
    Writes the trained model to the fiven file.
    '''
    pickle.dump(model,open(model_file, 'wb'))

def load_data(data_file):
    '''
    Loads X and Y data from the pickled data_file
    '''
    data = pickle.load(open(data_file)) 
    return (data['X'], data['Y'])

def get_adboost_classifier(algo, num_estimators, wl_loss, wl_penalty):
    '''
    Constructs a adaboost classifier object based on the algorithm, number of
    estimators, loss and penalty function given. Configures the object to run on
    all cores.
    '''
    weak_learner = SGDClassifier(loss=wl_loss, penalty=wl_penalty, verbose=True,
            n_jobs=-1)
    ab_classifier = AdaBoostClassifier( weak_learner, n_estimators =
                                        num_estimators, algorithm = algo)
    return ab_classifier

def train(ab_classifier, data_file, model_file):
    '''
    Takes a configured adaboost classifier object and train it with the training
    data from the data_file and write the learned model to the model_file.
    '''
    train_x, train_y = load_data(data_file)
    ab_classifier = ab_classifier.fit(train_x, train_y)
    write(ab_classifier, model_file) 

def test(model_file, data_file):
    '''
    Tests the model on the test data in data_file using the model in model_file.
    Prints accuracy to report the performance of the classifier. 
    '''
    test_x, test_y = load_data(data_file)
    ab_classifier = pickle.load(open(model_file))
    pred_y = ab_classifier.predict(test_x)
    correct = np.count_nonzero(test_y == pred_y)
    print 'Accuracy: ', correct * 100 / (1.0 * len(test_y))

def parse_train_args(num_learners, loss, penalty, data_file, model_file):
    '''
    parsers args required for training and calls the appropriate function.
    '''
    ab_classifier = get_adboost_classifier('SAMME', num_learners, loss, 
            penalty)
    train(ab_classifier, data_file, model_file)

def parse_test_args(model_file, data_file):
    '''
    parsers args required for testing and calls the appropriate function.
    '''
    test(args.model_file, args.data_file)

if __name__ == '__main__':
    parser = argparse.ArgumentParser() 
    parser.add_argument('data_file', help='path to the file containing training\
                        or testing data')
    parser.add_argument('model_file', help='path to the model file') 
    parser.add_argument('num_learners', nargs='?', help='number of boosting\
            rounds', default = 10, type=int)

    command_gp = parser.add_mutually_exclusive_group()
    command_gp.set_defaults(cmd = 'train')
    command_gp.add_argument('--train', action = 'store_const', dest = 'cmd',
            const = 'train', help = 'train adaboost SAMME model')
    command_gp.add_argument('--test', action = 'store_const', dest = 'cmd',
            const = 'test', help = 'predict on test data using given adaboost\
            SAMME model')

    loss_gp = parser.add_mutually_exclusive_group()
    loss_gp.set_defaults(loss = 'log')
    loss_gp.add_argument('--log_loss', action = 'store_const', dest = 'loss',
            const = 'log', help = 'use log loss function for training weak\
            learners')
    loss_gp.add_argument('--hinge_loss', action = 'store_const', dest = 'loss',
            const = 'hinge', help = 'use hinge loss function for training weak\
            learners')

    penalty_gp = parser.add_mutually_exclusive_group()
    penalty_gp.set_defaults(pen = 'l1')
    penalty_gp.add_argument('--l1', action = 'store_const', dest = 'pen', const
            = 'l1', help = 'use l1 penalty for training weak learners')
    penalty_gp.add_argument('--l2', action = 'store_const', dest = 'pen', const
            = 'l2', help = 'use l2 penalty for training weak learners')

    args = parser.parse_args()
    if args.cmd == 'train': parse_train_args(args.num_learners, args.loss,
            args.pen, args.data_file, args.model_file)
    elif args.cmd == 'test': parse_test_args(args.model_file, args.data_file)

