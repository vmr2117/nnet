import argparse
import numpy as np

from database import DatabaseAccessor
from data_structures import Distribution
from itertools import groupby
from operator import itemgetter
from pylab import *


def graph(db_file, key, filename, ttl):
    db = DatabaseAccessor(Distribution, db_file)
    data = [item for item in db.read() if item[0]==key]
    save_fig(data,key,  filename, ttl)

def save_fig(perf_data, key, filename, ttl):
    colors = ['blue','yellow','green', 'red']
    perf_data.sort(key = lambda tup: tup[2])
    i = 0
    fig, ax = plt.subplots(1)
    print perf_data
    for k, g in groupby(perf_data, itemgetter(2)):
        data = [(item[3], item[4]) for item in g]
        mean, std = map(list, zip(*data))
        mean = np.array(mean)
        std = np.array(std)
        ax.plot(range(len(data)), mean, lw = 2, label= 'layer '+str(i), color =
                colors[i % len(colors)])
        ax.fill_between(range(len(data)), mean + std, mean - std,
                facecolor=colors[i % len(colors)], alpha = 0.5)
        i += 1

    legend(loc = 'upper left')
    xlabel('Epoch')
    ylabel(key)
    title(ttl)
    grid(True)
    savefig(filename)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Tool to graph the validation \
                            and training error collected during training phase')
    parser.add_argument('database_file', help='the database file that contains \
                        data')
    parser.add_argument('key', help='key to identify the points from database')
    parser.add_argument('figure_file', help='graph of the data')
    parser.add_argument('title', help='graph title')
    args = parser.parse_args()
    graph(args.database_file, args.key, args.figure_file, args.title)
