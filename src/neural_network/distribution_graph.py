import argparse
import numpy as np
import os
import matplotlib.pyplot as plt

from database import DatabaseAccessor
from data_structures import Distribution
from itertools import groupby
from operator import itemgetter
from matplotlib.font_manager import FontProperties


def graph(db_file, filename_prefix):
    db = DatabaseAccessor(Distribution, db_file)
    data = [item for item in db.read()]
    save_fig(data, filename_prefix)

def save_fig(data, filename_prefix):
    colors = ['blue','green', 'red', 'black', 'magenta']
    # activations graph
    fig, ax = plt.subplots(1)
    p_activations = [item for item in data if item[0] == 'positive activations']
    p_activations.sort(key=lambda tup: tup[2])
    i = 0
    for k, g in groupby(p_activations, itemgetter(2)):
        _, _, _, mean, std, perc = map(np.array, zip(*g))
        x = range(len(mean))
        ax.errorbar(x, mean, yerr = std, color = colors[i % len(colors)])
        ax.plot(x, perc, '^', color = colors[i % len(colors)])
        ax.axhline(y=1.7159, color = 'red')
        i += 1

    n_activations = [item for item in data if item[0] == 'negative activations']
    n_activations.sort(key=lambda tup: tup[2])
    i = 0
    for k, g in groupby(n_activations, itemgetter(2)):
        _, _, _, mean, std, perc = map(list, zip(*g))
        x = range(len(mean))
        ax.errorbar(x, mean, yerr = std, label='Layer '+str(i+1) +
        ' mean & std dev', color = colors[i % len(colors)])
        ax.plot(range(len(mean)), perc, '^', label= 'Layer ' +
                str(i+1) + ' 98 percentiles', color =colors[i % len(colors)])
        ax.axhline(y=-1.7159, color = 'red', label = 'Saturation value')
        i+=1

    ax.legend(loc = 'center right') 
    ax.set_xlabel('Epochs')
    ax.set_ylabel('Activations')
    ax.set_title('Activations over training epochs')
    ax.grid()
    plt.savefig(filename_prefix +'_activations.eps')

    # weights graph
    fig, ax = plt.subplots(1)
    p_weights = [item for item in data if item[0] == 'positive weights']
    p_weights.sort(key=lambda tup: tup[2])
    i = 0
    for k, g in groupby(p_weights, itemgetter(2)):
        _, _, _, mean, std, perc = map(np.array, zip(*g))
        x = range(len(mean))
        ax.plot(x , mean, color = colors[i % len(colors)])
        ax.plot(x , std, '^', color = colors[i % len(colors)])
        i += 1

    n_weights = [item for item in data if item[0] == 'negative weights']
    n_weights.sort(key=lambda tup: tup[2])
    i = 0
    for k, g in groupby(n_weights, itemgetter(2)):
        _, _, _, mean, std, perc = map(np.array, zip(*g))
        x = range(len(mean))
        ax.plot(x, mean, label= 'L'+str(i+1) + ' wt mean', color =
                colors[i % len(colors)])
        ax.plot(x, -1 * std, '^', label= 'L'+str(i+1) + ' wt std',
                color = colors[i % len(colors)])
        i += 1

    p_bias = [item for item in data if item[0] == 'positive bias']
    p_bias.sort(key=lambda tup: tup[2])
    i = 0
    for k, g in groupby(p_bias, itemgetter(2)):
        _, _, _, mean, std, perc = map(np.array, zip(*g))
        x = range(len(mean))
        ax.plot(x, mean, color = colors[(i+2) % len(colors)])
        ax.plot(x, std, '^',color = colors[(i+2) % len(colors)])
        i += 1

    n_bias = [item for item in data if item[0] == 'negative bias']
    n_bias.sort(key=lambda tup: tup[2])
    i = 0
    for k, g in groupby(n_bias, itemgetter(2)):
        _, _, _, mean, std, perc = map(np.array, zip(*g))
        x = range(len(mean))
        ax.plot(x, mean, label= 'L'+str(i+1) + ' bs mean', color =
                colors[(i+2) % len(colors)])
        ax.plot(x, -1 * std, '^',label= 'L'+str(i+1) + ' bs std', color =
                colors[(i+2) % len(colors)])
        i += 1
    box = ax.get_position()
    ax.set_position([box.x0, box.y0, box.width*0.8, box.height])
    ax.legend(loc='center left', bbox_to_anchor=(1, 0.5)) 
    ax.set_xlabel('Epochs')
    ax.set_ylabel('Weights')
    ax.set_title('Weights over training epochs')
    ax.grid()
    plt.savefig(filename_prefix+'_weights.eps')

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Graph network weights and \
                        activation distributions over training epochs')
    parser.add_argument('database_file', help='the database file that contains \
                        the weight and activation distributions')
    parser.add_argument('filename_prefix', help='filename prefix for the \
                        output plots')
    args = parser.parse_args()
    graph(args.database_file, args.filename_prefix)
