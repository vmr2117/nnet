'''
Script to train a neural network. See command line options for more details.
'''
import argparse
import cPickle as pickle
import numpy as np
import sys

from db_interface import db_interface
from neural_network import network
from pylab import *
from signal import signal
import time


def train(args):
    s = time.time()
    tr_data = pickle.load(open(args.train_file))
    vd_data = pickle.load(open(args.validation_file))
    init_wts = None
    hidden_units = None
    if args.init_wt_file and not args.save_init_wt:
        init_wts = pickle.load(open(args.init_wt_file))
    if args.hidden_units: hidden_units = args.hidden_units
    
    nnet = network(args.actv)
    db = db_interface(args.model_perf_db)
    db.create_table()
    init_theta, theta = nnet.train(db, tr_data['X'],
            tr_data['Y'], vd_data['X'], vd_data['Y'], args.hidden_units,
            init_wts, args.mini_batch_size, args.epochs, args.validation_freq)
    print 'Training time:', time.time() - s, 'seconds'
    pickle.dump(theta, open(args.model_file, 'wb'))
    if args.save_init_wt and args.init_wt_file:
        pickle.dump(init_theta, open(args.init_wt_file, 'wb'))

def test(args):
    nnet = network(args.actv)
    theta = pickle.load(open(args.model_file))
    data = pickle.load(open(args.test_file))
    print "Accuracy :", 1.0 - nnet.test(data['X'], data['Y'], theta)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(help = 'sub-command help')
    train_parser = subparsers.add_parser('train', help= 'train neural network')
    train_parser.add_argument('train_file', help='path to training data')
    train_parser.add_argument('validation_file', help='path to validation data')
    train_parser.add_argument('model_file', help='filepath for model')
    train_parser.add_argument('model_perf_db', help='filepath for a file db \
            where training and validation errors are stored')
    train_parser.add_argument('epochs', help='number of epochs to train', type=int)
    train_parser.add_argument('validation_freq', help='frequency of validation', type=int)
    train_parser.add_argument('mini_batch_size', help='mini_batch size for SGD', type=int)
    train_parser.add_argument('hidden_units', nargs='?', help='number of hidden \
            units', type = int) 
    train_parser.add_argument('init_wt_file', nargs='?', help='path to the file \
            containing weights to initialize network.')
    train_parser.add_argument('--save_init_wt', action='store_true', help='save initial \
            wts generated to the init_wt_file')
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

