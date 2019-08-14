
import sqlite3
from sqlite3 import Error
 
 
def create_connection(db_file):
    """ create a database connection to the SQLite database
        specified by the db_file
    :param db_file: database file
    :return: Connection object or None
    """
    try:
        conn = sqlite3.connect(db_file, check_same_thread=False)
        return conn
    except Error as e:
        print(e)
 
    return None


def get_fields(conn):
    cur = conn.cursor()
    cur.execute("PRAGMA table_info(product);")
    rows = cur.fetchall()
    
    headers = list(map(lambda x: x[1], rows))

    return headers


def get_product(conn, product_id):
    cur = conn.cursor()
    cur.execute("SELECT * FROM product WHERE product_id='%s'" % product_id)
 
    rows = cur.fetchall()
    rows = list(rows)
    if len(rows):
        return rows[0]
    
    return None

def random_sample(conn, num_of_samples=1):
    cur = conn.cursor()

    products = []
    cur.execute("SELECT product_id FROM product ORDER BY RANDOM() LIMIT %d;" % num_of_samples)

    products += cur.fetchall()
    
    return [x[0] for x in products]

if __name__ == "__main__":

    conn = create_connection('../db/tiki-products.db')
    for _ in range(50):
        print(random_sample(conn, 3))
    conn.close()