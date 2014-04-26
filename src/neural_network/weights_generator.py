import argparse
import cPickle as pickle
import numpy as np

def get_layer_weights(layer_units, activation_func):
    """Returns the list of weights mapping consecutive layers of the network.

    Parameters
    ----------
    layer_units : list(int)
        List specifying the number of units in the layers of the network. The
        first entry specifies the number of inputs and the last entry specifies
        the number of output units.

    activation_func : str
        Type of activation units used in the network - 'tanh' or 'logistic'

    Return
    ------
    theta : list(array_like), shape(n_units, n_inputs)
        List of weights mapping consecutive layers of the network.

    bias : list(array_like), shape(n_units)
        List of biases for hidden and output layers of the network.
    """
    wt = None
    theta = None
    if activation_func == 'tanh':
        wt = [np.sqrt(6.0/(layer_units[layer-1] + layer_units[layer]))
                for layer in range(1, len(layer_units))]
    elif activation_func == 'logistic':
        wt = [4*np.sqrt(6.0/(layer_units[layer-1] + layer_units[layer]))
                for layer in range(1, len(layer_units))]
        
    theta = [np.random.uniform(low=-wt[layer-1], high=wt[layer-1],
                               size=(layer_units[layer], 
                                     layer_units[layer-1]))
                    for layer in range(1, len(layer_units))]
    bias = [np.zeros(layer_units[layer]) 
                    for layer in range(1, len(layer_units))]
           
    return theta, bias
        
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description ='Generate initial random \
                                    weights for feed forward neural network \
                                    based on the activation functions and the \
                                    architecture.')
    parser.add_argument('weights_file', help='file to write weights', type=str)
    parser.add_argument('activation_func', help='activation function \
                        used',type=str)
    parser.add_argument('layer_units', nargs='+', help='list of number of \
                        units in each layer',type=int)

    args = parser.parse_args()
    theta, bias = get_layer_weights(args.layer_units, args.activation_func)
    pickle.dump([theta, bias], open(args.weights_file,'wb'))
