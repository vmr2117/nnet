import argparse, gzip, numpy
import copy
import cPickle as pickle
import numpy as np
import os

def create_data_set(file_path, destination_dir):
    """Loads mnist data and standardizes the features and saves the training,
    validation and test data.

    Parameters
    ----------
    filepath : string, file path
        Path to the mnist.pkl.gz file
    
    destination_dir : string, directory path
        Directory path for the result files.
    """
    f = gzip.open(file_path, 'rb')
    (tr_d, tr_l), (vd_d, vd_l), (ts_d, ts_l)= pickle.load(f)
    f.close()

    mean = tr_d.mean(axis=0)
    std = tr_d.std(axis=0)
    std[std == 0] = 1.0 # when std_dev is 0 set it to 1.

    tr_d -= mean
    tr_d /= std

    vd_d -= mean
    vd_d /= std

    ts_d -= mean
    ts_d /= std

    train_name = 'numpy_array_multiclass.train'
    valid_name = 'numpy_array_multiclass.valid'
    test_name = 'numpy_array_multiclass.test'
    pickle.dump({'X':tr_d, 'Y':tr_l},
                open(os.path.join(destination_dir, train_name), 'wb'))
    pickle.dump({'X':vd_d, 'Y':vd_l},
                open(os.path.join(destination_dir, valid_name), 'wb')) 
    pickle.dump({'X':ts_d, 'Y':ts_l},
                open(os.path.join(destination_dir, test_name), 'wb'))

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description = 'Standardize the mnist \
                                training, validation and testing data set and \
                                writes the result files.')
    parser.add_argument('mnist_pkl_path', help = 'path to mnist.pkl.gz')
    parser.add_argument('dest_dir', help = 'directory to store the output files')
    args = parser.parse_args()
    create_data_set(args.mnist_pkl_path, args.dest_dir)
