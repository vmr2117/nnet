'''
Script to train a neural network. See command line options for more details.
'''
import argparse
import cPickle as pickle
import numpy as np
import sys

from db_interface import db_interface
from feedfwd_neural_network import FFNeuralNetwork

def train(args):
    tr_data = pickle.load(open(args.train_file))
    vd_data = pickle.load(open(args.validation_file))
    [theta, bias] = pickle.load(open(args.init_wt_file))
    
    nnet = FFNeuralNetwork()
    nnet.set_activation_func(args.actv)
    nnet.set_output_func('softmax')
    nnet.initialize(theta, bias)
    db = db_interface(args.model_perf_db)
    db.create_table()
    btheta, bbias = nnet.train(db, tr_data['X'],
            tr_data['Y'], vd_data['X'], vd_data['Y'], args.mini_batch_size,
            args.epochs, args.validation_freq, args.learn_only_last)
    pickle.dump([btheta,bbias], open(args.model_file, 'wb'))

def test(args):
    [theta, bias] = pickle.load(open(args.model_file))
    nnet = FFNeuralNetwork()
    nnet.set_activation_func(args.actv)
    nnet.set_output_func('softmax')
    nnet.initialize(theta, bias)
    data = pickle.load(open(args.test_file))
    print 'Error :',nnet.test(data['X'], data['Y'])

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(help = 'sub-command help')
    train_parser = subparsers.add_parser('train', help= 'train neural network')
    train_parser.add_argument('train_file', help='path to training data')
    train_parser.add_argument('validation_file', help='path to validation data')
    train_parser.add_argument('init_wt_file', help='path to the file containing\
            weights to initialize network.')
    train_parser.add_argument('model_file', help='filepath for model')
    train_parser.add_argument('model_perf_db', help='filepath for a file db \
            where training and validation errors are stored')
    train_parser.add_argument('epochs', help='number of epochs to train', type=int)
    train_parser.add_argument('validation_freq', help='frequency of validation', type=int)
    train_parser.add_argument('mini_batch_size', help='mini_batch size for SGD', type=int)
    train_parser.add_argument('--learn_only_last', help='learn only last \
            layer', action='store_true')

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

