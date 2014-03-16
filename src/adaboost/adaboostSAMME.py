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

def load_traindata(data_file):
    '''
    Loads train data from the pickle data_file
    '''
    data = pickle.load(open(data_file)) 
    return (data['train_x'], data['train_y'])

def load_testdata(data_file):
    '''
    Loads test data from the pickle data_file
    '''
    data = pickle.load(open(data_file)) 
    return (data['test_x'], data['test_y'])

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
    train_x, train_y = load_traindata(data_file)
    ab_classifier = ab_classifier.fit(train_x, train_y)
    write(ab_classifier, model_file) 

def test(model_file, data_file):
    '''
    Tests the model on the test data in data_file using the model in model_file.
    Prints accuracy to report the performance of the classifier. 
    '''
    test_x, test_y = load_testdata(data_file)
    ab_classifier = pickle.load(open(model_file))
    pred_y = ab_classifier.predict(test_x)
    correct = np.count_nonzero(test_y == pred_y)
    print 'Accuracy: ', correct * 100 / (1.0 * len(test_y))

def parse_train_args(args):
    '''
    parsers args required for training and calls the appropriate function.
    '''
    ab_classifier = get_adboost_classifier(args.adb_algo, args.num_learners,
                                           args.loss_fn, args.penalty)
    train(ab_classifier, args.data_file, args.model_file)

def parse_test_args(args):
    '''
    parsers args required for testing and calls the appropriate function.
    '''
    test(args.model_file, args.data_file)

if __name__ == '__main__':
    parser = argparse.ArgumentParser() 
    parser.add_argument('data_file', help='train test data file')
    parser.add_argument('model_file', help='model file') 

    subparsers = parser.add_subparsers(title='subcommand')

    parser_train = subparsers.add_parser('train', help='train adaboost')
    parser_train.add_argument('loss_fn', help='weak learner loss function') 
    parser_train.add_argument('penalty', help='weak learner penalty') 
    parser_train.add_argument('num_learners', help='number of weak learners',
                                type=int) 
    parser_train.add_argument('adb_algo', help='adaboost algorithm to use') 
    parser_train.set_defaults(func=parse_train_args)

    parser_test = subparsers.add_parser('test', help='run classifier on test \
                                        data')
    parser_test.set_defaults(func=parse_test_args)
    args = parser.parse_args()
    args.func(args)

