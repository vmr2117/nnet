"""Data Structures for performance monitoring and debugging neural network.
These data structures are python types that can be used with sqlite3 databases.
"""
import sqlite3

"""Data structure for storing distribution
"""
class Distribution(object):
    name = 'dist'
    def __init__(self, str_id, iter_no, layer, mean, std, perc_98):
        """Initializer.

        Parameters
        ----------
        str_id : str
            String identifier for the data.

        iter_no : int
            Iteration number.

        layer : int
            Layer number.

        mean : float
            Mean of the distribution.

        std : float
            Standard deviation of the distribution.

        perc : float
            98'th perventile
        """
        self.str_id = unicode(str_id)
        self.iter_no = iter_no
        self.layer = layer
        self.mean = mean
        self.std = std
        self.perc = perc_98

    def __repr__(self):
        return '%s;%d;%d;%f;%f;%f' % (self.str_id, self.iter_no, self.layer,
                self.mean, self.std, self.perc)

    def get_tuple(self):
        """Returns the tuple of members in the object.
        
        Return
        ------
        ret : tuple(str_id, iter_no, layer, mean, std)
            Tuple of members in the object
        """
        ret = (self.str_id, self.iter_no, self.layer, self.mean, self.std,
                self.perc)
        return ret

    @staticmethod
    def adapt_point(dist):
        return '%s;%d;%d;%f;%f;%f' % (dist.str_id, dist.iter_no, dist.layer,
                dist.mean, dist.std, dist.perc)

    @staticmethod
    def convert_point(s):
        str_id, iter_no, layer, mean, std, perc = s.split(';')
        iter_no = int(iter_no)
        layer = int(layer)
        mean = float(mean)
        std = float(std)
        perc = float(perc)
        return Distribution(str_id, iter_no, layer, mean, std, perc)

"""Data structure for storing performance metrics.
"""
class Perf(object):
    name = 'perf'
    def __init__(self, it, tr_err, vd_err):
        """Initializer.

        Parameters
        ----------
        it : str
            String identifier for the data.

        tr_err : float
            Iteration number.

        vd_err : float
            Layer number.
        """
        self.it = it
        self.tr_err = tr_err
        self.vd_err = vd_err

    def __repr__(self):
        return '%d;%f;%f' % (self.it, self.tr_err, self.vd_err)

    def get_tuple(self):
        """Returns the tuple of members in the object.
        
        Return
        ------
        ret : tuple(it, tr_err, vd_err)
            Tuple of members in the object
        """

        return (self.it, self.tr_err, self.vd_err)

    @staticmethod
    def adapt_point(perf_data):
        return '%d;%f;%f' % (perf_data.it, perf_data.tr_err, perf_data.vd_err)

    @staticmethod
    def convert_point(s):
        it, tr_err, vd_err = s.split(';')
        it = int(it)
        tr_err = float(tr_err)
        vd_err = float(vd_err)
        return Perf(it, tr_err, vd_err)


