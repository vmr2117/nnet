'''
mnist_data.py script provides methods to fetch and massage mnist data as
required. Look at command line options for details.
'''
import argparse
import cPickle as pickle
import numpy as np
                        
from os import path
from sklearn.datasets import fetch_mldata
from sklearn.utils import shuffle

def fetch_mnistdata(train_pct, classes, indveclab):
    '''
    Fetches mnist digits dataset for the given classes and splits the data into
    training and testing sets according to the trainsize percentage. The first
    dictionary in the returned tuple is the training set and the second
    dictionary in the tuple is the testing set. If indveclab is true, then the
    labels are stored as indicator vectors.
    '''
    mnist = fetch_mldata('MNIST original')
    data = mnist.data * 1.0 / 255 # normalize the inputs.
    target = mnist.target.astype(int) # convert labels to int

    data = data[np.logical_or.reduce([target == cls for cls in classes])]  
    target = target[np.logical_or.reduce([target == cls for cls in classes])]
    data, target = shuffle(data, target, random_state=34)

    if indveclab == True:
        num_classes = np.unique(target)
        t = np.zeros((target.size, num_classes), dtype=int)
        t[np.array(range(t.size)), t] = 1
        target = t

    trainsize = int(len(target) * train_pct)
    return ({'X':data[:trainsize], 'Y':target[:trainsize]},   
            {'X':data[trainsize:], 'Y':target[trainsize:]}) 

def get_feat_string(data):
    return ' '.join([''.join([str(i),':',str(data[i])]) for i in range(data.size)
                                                if data[i] > 0 ])

def write_wv_format(data, file_handle):
    (r, _) = data['X'].shape
    for row in range(r):
        ex = ' '.join([str(data['Y'][row]), ' 1 ', '|',
                get_feat_string(data['X'][row,:]), '\n'])
        file_handle.write(ex)

def write(data, data_dir, fmt, filename_suff=''):
    '''
    Create train and test data files in the requested format fmt. Filename
    suffix is appended to the filenames; this should be used to identify the
    specifics of dataset like the number of classes etc. 
    '''
    tr_data = data[0]
    ts_data = data[1]
    if fmt == 'numpy_array':
        pickle.dump(tr_data,
                    open(path.join(data_dir, fmt + '_' + filename_suff + '.train'), 'wb'))
        pickle.dump(ts_data,
                    open(path.join(data_dir, fmt + '_' + filename_suff + '.test'), 'wb'))
    elif fmt == 'vw':
        write_wv_format(tr_data,
                    open(path.join(data_dir, fmt + '_' + filename_suff + '.train'), 'wb'))
        write_wv_format(ts_data,
                    open(path.join(data_dir, fmt + '_' + filename_suff + '.test'), 'wb'))
    else:
        print 'Unsupported format!' + fmt
        raise

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description = 'Create test and train files\
        from mnist dataset. If no options are specified create train and test\
        files for multiclass and store examples in dictionary of numpy arrays') 

    fmt_gp = parser.add_mutually_exclusive_group()
    fmt_gp.set_defaults(fmt = 'numpy_array')
    fmt_gp.add_argument('--vw_fmt', action = 'store_const', dest = 'fmt',
        const = 'vw', help = 'write train/test files in vowpal wobbit format')
    fmt_gp.add_argument('--numpy_array_fmt', action = 'store_const', dest = 'fmt',
        const = 'numpy_array', help = 'write train/test files as dicts of\
        numpy arrays')

    parser.add_argument('train_proportion', help='proportion of examples to include \
                        for training set. the remaining examples are used for \
                        testing set', type=float)
    parser.add_argument('path', help='directory to save train and test files') 

    cl_gp = parser.add_mutually_exclusive_group()
    cl_gp.set_defaults(cl_type = 'multiclass')
    cl_gp.add_argument('--multiclass', action='store_const', dest='cl_type',
        const='multiclass', help='for multiclass classification problem')
    cl_gp.add_argument('--binary', action='store_const', dest='cl_type',
        const='binary' , help='for binary classification problem')
    
    parser.add_argument('--indveclab', action='store_const', dest='indveclab',
        const='indicator',help='store class labels as indicator vectors')

    parser.add_argument('classes', nargs='?', default='89', help= 'in case of \
        two class specify the two classes in a string without any space') 

    args = parser.parse_args()

    if args.cl_type == 'binary' and (not args.classes or len(args.classes) !=2):
        parser.error("can't {} without a classes argument".format(args.mode)) 

    classes = [i for i in range(10)]
    suffix = 'multiclass'
    if args.cl_type == 'binary': 
        classes = [int(args.classes[0]), int(args.classes[1])]
        suffix = 'binary_' + args.classes

    indveclab = False
    if args.indveclab == 'indveclab': 
        indveclab = True
        suffix += '_indicator_target'
    data = fetch_mnistdata(args.train_proportion, classes, indveclab)
    write(data, args.path, args.fmt, suffix)

