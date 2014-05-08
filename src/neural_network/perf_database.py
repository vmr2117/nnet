"""PerfDatabase module implements method to create database, write and read
performance parameters of the models to disk.
"""
import sqlite3

class PerfDatabase:
    def __init__(self, file_path):
        """
        Parameters
        ----------
        file_path : str
            path of the file backing the database.
        """
        self.conn = sqlite3.connect(file_path)
        self.c = self.conn.cursor()
        self.table_name = 'model_perf_data'
        self.w_str_open = ('INSERT INTO ' + self.table_name
                        + ' values (' )
        self.w_str_close = ')'
        self.r_str= ('select * from ' + self.table_name)

    def create_table(self):
        """
        Creates 'model_perf_data' table. 
        """
        create_str = ('CREATE TABLE ' + self.table_name
                        + ' (iter integer, tr_err real, vd_err real)')
        self.c.execute(create_str)
        self.conn.commit()

    def write(self, it, tr_err, vd_err):
        """Writes the training and validation error to the database.

        Parameters
        ----------
        it : iteration number
            The iteration number.

        tr_err : error obtained on the training set.

        vd_err : error obtained on the validation set.
        """
        write_str = (self.w_str_open + ','.join([str(it), str(tr_err), str(vd_err)])
                     + self.w_str_close)
        self.c.execute(write_str)
        self.conn.commit()

    def read(self):
        """Returns the entire data from 'model_perf_data' table.

        Return
        ------
        perf_data : list(tuple)
            performance data.
        """
        perf_data = [row for row in self.c.execute(self.r_str)]
        return perf_data

