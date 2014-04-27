import argparse

from perf_database import PerfDatabase
from pylab import *


def graph(rand_db, adaboost_db, filename, ttl, x_lim, y_lim):
    db = PerfDatabase(rand_db)
    rand_data = db.read()
    db = db_interface(adaboost_db)
    adaboost_data = db.read()
    save_fig(rand_data, adaboost_data, filename, ttl, x_lim, y_lim)

def save_fig(rand_data, adaboost_data, filename, ttl, x_lim, y_lim):
    it1 = [ entry[0] for entry in rand_data]
    it2 = [ entry[0] for entry in adaboost_data]
    iters = min(len(it1), len(it2))
    x_axis = range(iters)
    rand_nnet_tr_errs = [ entry[1] for entry in rand_data[:iters]]
    rand_nnet_vd_errs = [ entry[2] for entry in rand_data[:iters]]
    adaboost_nnet_tr_errs = [ entry[1] for entry in adaboost_data[:iters]]
    adaboost_nnet_vd_errs = [ entry[2] for entry in adaboost_data[:iters]]
    plot(x_axis, rand_nnet_tr_errs, 'g', label = 'rand init Training Error')
    plot(x_axis, rand_nnet_vd_errs, 'r--', label = 'rand init Validation Error')
    plot(x_axis, adaboost_nnet_tr_errs, 'b', label = 'adaboost init Training Error')
    plot(x_axis, adaboost_nnet_vd_errs, 'k--', label = 'adaboost init Validation Error')
    if x_lim : xlim([0, x_lim])
    if y_lim : ylim([0, y_lim])
    else: ylim([0, 0.2])
    legend()
    xlabel('Epoch')
    ylabel('Error')
    title(ttl)
    grid(True)
    savefig(filename)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Tool to graph the validation \
                            and training error collected during training phase')
    parser.add_argument('database_file_nnet_rand', help='the database file that contains \
                        data for randomly initialized neural network')
    parser.add_argument('database_file_nnet_adaboost', help='the database file that contains \
                        data for adaboost initialized neural network')
    parser.add_argument('figure_file', help='graph of the data')
    parser.add_argument('title', help='figure title')
    parser.add_argument('x_lim', nargs='?', help='graph x high limit', type = float)
    parser.add_argument('y_lim', nargs='?', help='graph y high limit', type = float)
    args = parser.parse_args()
    x_lim = None
    y_lim = None
    if args.x_lim : x_lim = args.x_lim
    if args.y_lim : y_lim = args.y_lim
    graph(args.database_file_nnet_rand, args.database_file_nnet_adaboost,
            args.figure_file, args.title, x_lim, y_lim)
