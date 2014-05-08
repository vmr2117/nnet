import argparse, gzip, numpy
import copy
import cPickle as pickle
import numpy as np
import os

def get_vw_line(data, label, wt = None):
    l = str(int(label) + 1)
    if wt: l = l + ' ' + str(wt)
    feat = [ str(i+1)+':'+ str(data[i]) for i in range(data.size) 
            if data[i] != 0.0]
    return l +' | '+' '.join(feat) + '\n'


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

    train_name = os.path.join(destination_dir, 'vw_multiclass.train')
    valid_name = os.path.join(destination_dir,'vw_multiclass.valid')
    test_name = os.path.join(destination_dir, 'vw_multiclass.test')
    with  open(train_name, 'w') as outfile:
        for i in range(tr_l.size):
            outfile.write(get_vw_line(tr_d[i], tr_l[i], 1.0/50000))
    with  open(valid_name, 'w') as outfile:
        for i in range(vd_l.size):
            outfile.write(get_vw_line(vd_d[i], vd_l[i]))
    with  open(test_name, 'w') as outfile:
        for i in range(ts_l.size):
            outfile.write(get_vw_line(ts_d[i], ts_l[i]))

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description = 'Standardize the mnist \
                                training, validation and testing data set and \
                                writes the result files.')
    parser.add_argument('mnist_pkl_path', help = 'path to mnist.pkl.gz')
    parser.add_argument('dest_dir', help = 'directory to store the output files')
    args = parser.parse_args()
    create_data_set(args.mnist_pkl_path, args.dest_dir)
