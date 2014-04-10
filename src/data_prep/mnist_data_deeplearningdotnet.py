import argparse, gzip, numpy
import cPickle as pickle
import os

def create_data_set(filepath, destination_dir):
    ''' loads mnist.pkl.gz and write them into a format that is suitable for
    training and testing with the existing code.
    '''
    f = gzip.open(filepath, 'rb')
    train_set, valid_set, test_set = pickle.load(f)
    f.close()
    train_name = 'numpy_array_multiclass.train'
    valid_name = 'numpy_array_multiclass.valid'
    test_name = 'numpy_array_multiclass.test'
    pickle.dump({'X':train_set[0], 'Y':train_set[1]},
                open(os.path.join(destination_dir, train_name), 'wb'))
    pickle.dump({'X':valid_set[0], 'Y':valid_set[1]},
                open(os.path.join(destination_dir, valid_name), 'wb')) 
    pickle.dump({'X':test_set[0], 'Y':test_set[1]},
                open(os.path.join(destination_dir, test_name), 'wb'))

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description = 'Write mnist data from \
                                     deepleraning.net in the format that is \
                                     suitable for training')
    parser.add_argument('mnist_pkl_path', help = 'path to mnist.pkl.gz')
    parser.add_argument('dest_dir', help = 'directory to store the output files')
    args = parser.parse_args()
    create_data_set(args.mnist_pkl_path, args.dest_dir)
