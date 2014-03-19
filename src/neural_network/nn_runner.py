'''
Script to train a neural network. See command line options for more details.
'''
import argparse

from neural_network import train, predict, save_model, load_model

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
            const = 'test', help = 'predict on test data using given nnet model'

    actv_gp = parser.add_mutually_exclusive_group()
    actv_gp.set_defaults(actv = 'logistic')
    actv_gp.add_argument('--logistic_actv', action = 'store_const', dest =
            'actv', const = 'logistic', help = 'logistic activation function')
    actv_gp.add_argument('--tanh_actv', action = 'store_const', dest = 'actv',
            const = 'tanh', help = 'tanh activation function')

    penalty_gp = parser.add_mutually_exclusive_group()
    penalty_gp.set_defaults(pen = 'l2')
    penalty_gp.add_argument('--l1', action = 'store_const', dest = 'pen', const
            = 'l1', help = 'use l1 penalty')
    penalty_gp.add_argument('--l2', action = 'store_const', dest = 'pen', const
            = 'l2', help = 'use l2 penalty')

    args = parser.parse_args()
    if args.cmd == 'train': parse_train_args(args.num_learners, args.loss,
            args.pen, args.data_file, args.model_file)
    elif args.cmd == 'test': parse_test_args(args.model_file, args.data_file)

