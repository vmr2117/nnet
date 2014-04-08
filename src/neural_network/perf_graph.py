import argparse

from db_interface import db_interface
from pylab import *


def graph(db_file, filename = None):
    db = db_interface(db_file)
    perf_data = db.read()
    save_fig(perf_data, filename)

def save_fig(perf_data, filename = None):
    iters = [ entry[0] for entry in perf_data]
    tr_errs = [ entry[1] for entry in perf_data]
    vd_errs = [ entry[2] for entry in perf_data]
    plot(iters, tr_errs, 'g', label = 'Training Error')
    plot(iters, vd_errs, 'r', label = 'Testing Error')
    legend()
    xlabel('Numer of minibatches')
    ylabel('Error')
    title('Neural Network')
    grid(True)
    if filename: savefig(filename)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Tool to graph the validation \
                            and training error collected during training phase')
    parser.add_argument('database_file', help='the database file that contains \
                        data')
    parser.add_argument('figure_file', help='graph of the data')
    args = parser.parse_args()
    graph(args.database_file, args.figure_file)
