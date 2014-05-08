'''
adaboost.py script can be used to train and test adaboost with a SGDclassifier
for mnist data. Look up commandline options for more information.
'''
import argparse
import cPickle as pickle
import numpy as np
import pylab as pl
import time
    
from sklearn.ensemble import AdaBoostClassifier
from sklearn.linear_model import SGDClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score
from sklearn.metrics import zero_one_loss

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

def get_adboost_classifier(algo, num_estimators, wl_loss, wl_penalty, passes):
    '''
    Constructs a adaboost classifier object based on the algorithm, number of
    estimators, loss and penalty function given. Configures the object to run on
    all cores.
    '''
    '''
    weak_learner = SGDClassifier(loss=wl_loss, penalty=wl_penalty,
            n_jobs=-1, n_iter = passes, shuffle = True)
    '''
    weak_learner = DecisionTreeClassifier(max_depth=30)
    ab_classifier = AdaBoostClassifier( weak_learner, n_estimators =
                                        num_estimators, algorithm = algo)

    return ab_classifier

def train(ab_classifier, train_file, validation_file, model_file, graph_file):
    '''
    Takes a configured adaboost classifier object and train it with the training
    data from the data_file and write the learned model to the model_file.
    '''
    s = time.time()
    train_x, train_y = load_data(train_file)
    ab_classifier = ab_classifier.fit(train_x, train_y)
    write(ab_classifier, model_file) 
    valid_x, valid_y = load_data(validation_file)
    # find out stage wise training error
    n_estimators = len(ab_classifier.estimators_)
    train_err = np.zeros((n_estimators,))
    valid_err = np.zeros((n_estimators,))
    for i, y_pred in enumerate(ab_classifier.staged_predict(train_x)):
        train_err[i] = zero_one_loss(y_pred, train_y)
    for i, y_pred in enumerate(ab_classifier.staged_predict(valid_x)):
        valid_err[i] = zero_one_loss(y_pred, valid_y)
    save_fig(train_err, valid_err, n_estimators, graph_file)
    print 'Training time:', time.time() - s, 'seconds'

def save_fig(train_err, valid_err, n_estimators, file_name):
    fig = pl.figure()
    ax = fig.add_subplot(111)
    ax.plot(np.arange(n_estimators) + 1, train_err, label='Train Error', color='red')
    ax.plot(np.arange(n_estimators) + 1, valid_err, label='Validation Error',
            color='green')
    ax.set_ylim((0.0, 1.0))
    ax.set_xlabel('Number of Learners')
    ax.set_ylabel('Error')
    ax.set_title('Adaboost SAMME on MNIST dataset')
    ax.xaxis.grid(True)
    ax.yaxis.grid(True)
    leg = ax.legend(loc='upper right', fancybox=True)
    leg.get_frame().set_alpha(0.7)
    pl.savefig(file_name)

def test(model_file, test_file):
    '''
    Tests the model on the test data in data_file using the model in model_file.
    Prints accuracy to report the performance of the classifier. 
    '''
    test_x, test_y = load_data(test_file)
    ab_classifier = pickle.load(open(model_file))
    pred_y = ab_classifier.predict(test_x)
    correct = np.count_nonzero(test_y == pred_y)
    print 'Accuracy: ', correct / (1.0 * len(test_y))

def parse_train_args(args):
    '''
    parsers args required for training and calls the appropriate function.
    '''
    ab_classifier = get_adboost_classifier('SAMME.R', args.num_learners,
            args.loss, args.pen, args.epochs)
    train(ab_classifier, args.train_file, args.validation_file, args.model_file,
            args.graph_file)

def parse_test_args(args):
    '''
    parsers args required for testing and calls the appropriate function.
    '''
    test(args.model_file, args.test_file)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(help = 'sub-command help')
    train_parser = subparsers.add_parser('train', help= 'train adaboost')
    train_parser.add_argument('train_file', help='path to training data')
    train_parser.add_argument('validation_file', help='path to validation data')
    train_parser.add_argument('model_file', help='filepath for model')
    train_parser.add_argument('graph_file', help='filepath for training graph')
    train_parser.add_argument('epochs', help='number of epochs for weak \
            learners', type = int)
    train_parser.add_argument('num_learners', nargs='?', help='number of weak \
            learners', default = 10, type=int)
    loss_gp = train_parser.add_mutually_exclusive_group()
    loss_gp.set_defaults(loss = 'log')
    loss_gp.add_argument('--log_loss', action = 'store_const', dest = 'loss',
            const = 'log', help = 'use log loss function for training weak\
            learners')
    loss_gp.add_argument('--hinge_loss', action = 'store_const', dest = 'loss',
            const = 'hinge', help = 'use hinge loss function for training weak\
            learners')

    penalty_gp = train_parser.add_mutually_exclusive_group()
    penalty_gp.set_defaults(pen = 'l2')
    penalty_gp.add_argument('--l1', action = 'store_const', dest = 'pen', const
            = 'l1', help = 'use l1 penalty for training weak learners')
    penalty_gp.add_argument('--l2', action = 'store_const', dest = 'pen', const
            = 'l2', help = 'use l2 penalty for training weak learners')

    train_parser.set_defaults(func = parse_train_args)

    test_parser = subparsers.add_parser('test', help = 'test neural network')
    test_parser.add_argument('test_file', help='path to test data')
    test_parser.add_argument('model_file', help='filepath for model')
    test_parser.set_defaults(func = parse_test_args)

    args = parser.parse_args()
    args.func(args)



