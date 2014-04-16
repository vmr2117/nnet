import argparse
import cPickle as pickle
import numpy as N
import pylab as P

def _blob(x,y,area,colour):
    """
    Draws a square-shaped blob with the given area (< 1) at
    the given coordinates.
    """
    hs = N.sqrt(area) / 2
    xcorners = N.array([x - hs, x + hs, x + hs, x - hs])
    ycorners = N.array([y - hs, y - hs, y + hs, y + hs])
    P.fill(xcorners, ycorners, colour, edgecolor=colour)

def hinton(W, maxWeight=None):
    """
    Draws a Hinton diagram for visualizing a weight matrix. 
    Temporarily disables matplotlib interactive mode if it is on, 
    otherwise this takes forever.
    """
    reenable = False
    if P.isinteractive():
        P.ioff()
    height, width = W.shape
    if not maxWeight:
        maxWeight = 2**N.ceil(N.log(N.max(N.abs(W)))/N.log(2))

    P.fill(N.array([0,width,width,0]),N.array([0,0,height,height]),'gray')
    P.axis('off')
    P.axis('equal')
    for x in xrange(width):
        for y in xrange(height):
            _x = x+1
            _y = y+1
            w = W[y,x]
            if w > 0:
                _blob(_x - 0.5, height - _y + 0.5, min(1,w/maxWeight),'white')
            elif w < 0:
                _blob(_x - 0.5, height - _y + 0.5, min(1,-w/maxWeight),'black')
    if reenable:
        P.ion()

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
        sz = wts_1.size
        inds = N.random.randint(0, high=sz-1,size=(400,))
        P.subplot(2, 2, ind*2 + 1)
        hinton(wts_1.flatten()[inds].reshape((20,20)))
        P.title('Initial Weights: Layer '+ str(ind+1))
        P.subplot(2, 2, ind*2 + 2)
        hinton(wts_2.flatten()[inds].reshape((20,20)))
        P.title('Final Weights: Layer '+ str(ind+1))
    P.suptitle(args.title)
    P.savefig(args.image_path)



