'''
Script to train a neural network. See command line options for more details.
'''
import argparse
import cPickle as pickle
import numpy as np
import sys

from neural_network import network
from pylab import *

def train(data_file, actv, n_hidden_units, model_file, graph_figure_file,
        init_wts = None): 
    nnet = network(actv)
    data = pickle.load(open(data_file))
    cost_err, theta = nnet.train(data['X'], data['Y'], n_hidden_units, init_wts)
    pickle.dump(theta, open(model_file, 'wb'))
    save_fig(cost_err, graph_figure_file)

def save_fig(cost_err, file_name):
    print cost_err.keys()
    print [cost_err[key][0] for key in cost_err.keys()]
    print [cost_err[key][1] for key in cost_err.keys()]

    plot(cost_err.keys(), [cost_err[key][0] for key in cost_err.keys()], 'g^',
            label = 'Training Cost')
    plot(cost_err.keys(), [cost_err[key][1] for key in cost_err.keys()], 'ro',
            label = 'Validation Error')
    xlabel('Iteration')
    ylabel('Error and Cost')
    title('Training cost and Validation Error')
    legend()
    grid(True)
    savefig(file_name)
    show()

def test(data_file, actv, model_file):
    nnet = network(actv)
    theta = pickle.load(open(model_file))
    data = pickle.load(open(data_file))
    print "Accuracy :", nnet.evaluate(data['X'], data['Y'], theta)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('data_file', help='path to the file containing training\
                        or testing data')
    parser.add_argument('model_file', help='path to the model file')

    command_gp = parser.add_mutually_exclusive_group()
    command_gp.set_defaults(cmd = 'train')
    command_gp.add_argument('--train', action = 'store_const', dest = 'cmd',
            const = 'train', help = 'train neural network')
    command_gp.add_argument('--test', action = 'store_const', dest = 'cmd',
            const = 'test', help = 'predict on test data using given nnet \
            model')

    actv_gp = parser.add_mutually_exclusive_group()
    actv_gp.set_defaults(actv = 'logistic')
    actv_gp.add_argument('--logistic_actv', action = 'store_const', dest =
            'actv', const = 'logistic', help = 'logistic activation function')
    actv_gp.add_argument('--tanh_actv', action = 'store_const', dest = 'actv',
            const = 'tanh', help = 'tanh activation function')
    
    parser.add_argument('hidden_units', nargs='?', help='number of hidden \
            units', type = int) 
    parser.add_argument('init_wt_file', nargs='?', help='path to the file \
            containing initial weights.')

    args = parser.parse_args()

    if not args.hidden_units and args.cmd == 'train':
        print "No hidden_units provided"
        sys.exit(1)

    init_wts = None
    if args.init_wt_file: init_wts = pickle.load(open(args.init_wt_file))
    if args.cmd == 'train':
        train(args.data_file, args.actv, args.hidden_units, args.model_file,
                'train.png', init_wts)
    elif args.cmd == 'test':
        test(args.data_file, args.actv, args.model_file)

