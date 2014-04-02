'''
Script to train a neural network. See command line options for more details.
'''
import argparse
import cPickle as pickle
import numpy as np
import sys

from neural_network import network
from pylab import *

def train(args):
    nnet = network(args.actv)
    train_data = pickle.load(open(args.train_file))
    valid_data = pickle.load(open(args.validation_file))
    init_wts = None
    hidden_units = None
    if args.init_wt_file: init_wts = pickle.load(open(args.init_wt_file))
    if args.hidden_units: hidden_units = args.hidden_units
    cost_err, theta = nnet.train(train_data['X'], train_data['Y'],
                                 valid_data['X'], valid_data['Y'], 
                                 args.hidden_units, init_wts)
    pickle.dump(theta, open(args.model_file, 'wb'))
    save_fig(cost_err, args.graph_file)

def save_fig(cost_err, file_name):
    keys = list(sorted(cost_err.iterkeys()))
    plot(keys, [cost_err[key][0] for key in keys], 'g',
            label = 'Training Error')
    plot(keys, [cost_err[key][1] for key in keys], 'r',
            label = 'Validation Error')
    legend()
    xlabel('Iteration')
    ylabel('Error')
    title('Neural Network (500 hidden units) - random initalized, LR - 0.01')
    grid(True)
    savefig(file_name)
    show()

def test(args):
    nnet = network(args.actv)
    theta = pickle.load(open(args.model_file))
    data = pickle.load(open(args.test_file))
    print "Accuracy :", nnet.evaluate(data['X'], data['Y'], theta)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(help = 'sub-command help')
    train_parser = subparsers.add_parser('train', help= 'train neural network')
    train_parser.add_argument('train_file', help='path to training data')
    train_parser.add_argument('validation_file', help='path to validation data')
    train_parser.add_argument('model_file', help='filepath for model')
    train_parser.add_argument('graph_file', help='filepath for training graph')
    train_parser.add_argument('hidden_units', nargs='?', help='number of hidden \
            units', type = int) 
    train_parser.add_argument('init_wt_file', nargs='?', help='path to the file \
            containing initial weights.')
    actv_gp = train_parser.add_mutually_exclusive_group()
    actv_gp.set_defaults(actv = 'logistic')
    actv_gp.add_argument('--logistic_actv', action = 'store_const', dest =
            'actv', const = 'logistic', help = 'logistic activation function')
    actv_gp.add_argument('--tanh_actv', action = 'store_const', dest = 'actv',
            const = 'tanh', help = 'tanh activation function')
    train_parser.set_defaults(func = train)

    test_parser = subparsers.add_parser('test', help = 'test neural network')
    test_parser.add_argument('test_file', help='path to test data')
    test_parser.add_argument('model_file', help='filepath for model')
    actv_gp = test_parser.add_mutually_exclusive_group()
    actv_gp.set_defaults(actv = 'logistic')
    actv_gp.add_argument('--logistic_actv', action = 'store_const', dest =
            'actv', const = 'logistic', help = 'logistic activation function')
    actv_gp.add_argument('--tanh_actv', action = 'store_const', dest = 'actv',
            const = 'tanh', help = 'tanh activation function')
    test_parser.set_defaults(func = test)

    args = parser.parse_args()
    args.func(args)

