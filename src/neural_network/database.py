""" Implements Database reads and writes.
"""
import sqlite3

class DatabaseAccessor:
    def __init__(self, data_class, file_path):
        """Database Accessor class.

        Parameters
        ----------
        data_class : class
            Class for the python type to be stored in database.

        file_path : str
            Path to the file backing the database.
        """
        self.dname = data_class.name
        sqlite3.register_adapter(data_class, data_class.adapt_point)        
        sqlite3.register_adapter(self.dname, data_class.convert_point)        
        self.con = sqlite3.connect(file_path,
                                   detect_types=sqlite3.PARSE_DECLTYPES)

    def create_table(self):
        """Creates new table
        """
        crt_cmd = 'create table tab(point '+ self.dname + ')'
        self.con.execute(crt_cmd) 

    def write(self, point):
        """Writes point to the database.

        Parameters
        ----------
        point : object, type(data_class)
            Object to write to database.
        """
        inst_cmd = 'insert into tab(point) values (?)'
        self.con.execute(inst_cmd, (point,))

    def read(self):
        """Reads the entire table from the database.

        Return
        ------
        points_list : list(tuple)
            List of row entries from the table.
        """
        read_cmd = 'select point from tab'
        points_list = [row.get_tuple() for row in self.con.execute(read_cmd)]
        return points_list
