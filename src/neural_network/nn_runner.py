'''
Script to train a neural network. See command line options for more details.
'''
import argparse
import cPickle as pickle

from neural_network import train, predict, save_model, load_model

def train(data_file, layer_units, actv, init_wt_file, model_path):
    nnet = network(layer_units, actv)
    data = pickle.load(open(data_file))
    init_wts = pickle.load(open(init_wt_file))
    nnet.train(data['X'], data['Y'], init_wts)
    nnet.save_model(model_path)

def test(data_file, layer_units, actv, model_file):
    nnet = network(layer_units, actv)
    nnet.load_model(model_file))
    data = pickle.load(open(data_file))
    pred_y = nnet.predict(data['X'])
    print np.sum(pred_y == data['Y']) * 1.0 / data['Y'].size


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
    
    '''
    penalty_gp = parser.add_mutually_exclusive_group()
    penalty_gp.set_defaults(pen = 'l2')
    penalty_gp.add_argument('--l1', action = 'store_const', dest = 'pen', const
            = 'l1', help = 'use l1 penalty')
    penalty_gp.add_argument('--l2', action = 'store_const', dest = 'pen', const
            = 'l2', help = 'use l2 penalty')
    '''

    parser.add_argument('net_arch', nargs='?', help='comma separated values \
        indicating the number of units in each layer. The layers are specified \
        from left to right')
    parser.add_argument('init_wt_file', nargs='?', help='path to the file \
        containing initial weights.')

    args = parser.parse_args()

    assert args.net_arch, 'Network architecture not provided'
    layer_units = [int(num) for num in args.net_arch.strip().split(',')]
    if args.cmd == 'train':
        train(args.data_file, layer_units, args.actv, args.init_wt_file,
            args.model_file)
    elif args.cmd == 'test':
        test(args.data_file, layer_units, args.actv, args.model_file)

