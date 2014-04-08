'''
dw_writer class can be used to save the model validation data to a file backed
database on disk. This allows one to check the model performance asynchronously.  
'''
import sqlite3

class db_interface:
    def __init__(self, file_name):
        '''
        'file_name' is the name of the file to use for storing database. The
        table name is 'model_perf_data'.
        '''
        self.conn = sqlite3.connect(file_name)
        self.c = self.conn.cursor()
        self.table_name = 'model_perf_data'
        self.w_str_open = ('INSERT INTO ' + self.table_name
                        + ' values (' )
        self.w_str_close = ')'
        self.r_str= ('select * from ' + self.table_name)

    def create_table(self):
        # create table
        create_str = ('CREATE TABLE ' + self.table_name
                        + ' (iter integer, tr_err real, vd_err real)')
        self.c.execute(create_str)
        self.conn.commit()

    def write(self, it, tr_err, vd_err):
        '''
        Writes the iteration number 'it', validation error 'vd_err' and training
        error 'tr_err' to the database.
        '''
        write_str = (self.w_str_open + ','.join([str(it), str(tr_err), str(vd_err)])
                     + self.w_str_close)
        self.c.execute(write_str)
        self.conn.commit()

    def read(self):
        return [row for row in self.c.execute(self.r_str)]

