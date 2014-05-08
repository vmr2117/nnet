import argparse

from database import DatabaseAccessor
from data_structures import Perf
from pylab import *


def graph(db_file, filename, ttl):
    db = DatabaseAccessor(Perf, db_file)
    perf_data = db.read()
    save_fig(perf_data, filename, ttl)

def save_fig(perf_data, filename, ttl):
    iters = range(len(perf_data)) # number of epochs
    tr_errs = [ entry[1] for entry in perf_data]
    vd_errs = [ entry[2] for entry in perf_data]
    plot(iters, tr_errs, 'g', label = 'Training Error')
    plot(iters, vd_errs, 'r', label = 'Validation Error')
    legend()
    xlabel('Epoch')
    ylabel('Error')
    title(ttl)
    grid(True)
    savefig(filename)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Tool to graph the validation \
                            and training error collected during training phase')
    parser.add_argument('database_file', help='the database file that contains \
                        data')
    parser.add_argument('figure_file', help='graph of the data')
    parser.add_argument('title', help='graph title')
    args = parser.parse_args()
    graph(args.database_file, args.figure_file, args.title)
