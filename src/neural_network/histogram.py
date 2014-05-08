import argparse
import cPickle as pickle
import numpy as N
import pylab as P

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Compare layerwise weights of \
                                    two neural network models using hinton \
                                    diagrams. Compares corresponding weights \
                                    at sampled indices.')
    parser.add_argument('weights_1', help='pickle file containing neural network \
                                          model 1')
    parser.add_argument('weights_2', help='pickle file containing neural network \
                                          model 2')
    parser.add_argument('title', help='title for the comparison plot')
    parser.add_argument('image_path', help='path format to output images')
    args = parser.parse_args()

    model_1 = pickle.load(open(args.weights_1, 'rb'))
    model_2 = pickle.load(open(args.weights_2, 'rb'))
    fig = P.figure()
    for ind, (wts_1, wts_2) in enumerate(zip(model_1, model_2)):
        P.subplot(2, 2, ind*2 + 1)
        P.hist(wts_1.flatten())
        P.title('Initial Weights: Layer '+ str(ind+1))
        P.subplot(2, 2, ind*2 + 2)
        P.hist(wts_2.flatten())
        P.title('Final Weights: Layer '+ str(ind+1))
    P.suptitle(args.title)
    P.savefig(args.image_path)
