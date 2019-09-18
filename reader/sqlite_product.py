
import sqlite3
from sqlite3 import Error
 
 
def create_connection(db_file):
    """ create a database connection to the SQLite database
        specified by the db_file
    :param db_file: database file
    :return: Connection object or None
    """
    try:
        conn = sqlite3.connect(db_file, timeout=8, check_same_thread=False)
        return conn
    except Error as e:
        print(e)
 
    return None


def get_fields(conn, table_name="product"):
    cur = conn.cursor()
    cur.execute("PRAGMA table_info(%s);" % table_name)
    rows = cur.fetchall()
    
    headers = list(map(lambda x: x[1], rows))
    cur.close()
    return headers


def get_product(conn, product_id, table_name="product"):
    cur = conn.cursor()
    cur.execute("SELECT * FROM %s WHERE product_id='%s'" % (table_name, product_id))
 
    rows = cur.fetchall()
    rows = list(rows)
    cur.close()
    if len(rows):
        return rows[0]
    return None


def get_all_product_ids(conn, table_name="product"):
    cur = conn.cursor()
    cur.execute("SELECT product_id FROM %s" % table_name)
 
    rows = cur.fetchall()
    rows = [r[0] for r in list(rows)]
    cur.close()
    return rows


def get_batch_product_ids(conn, limit, offset):
    cur = conn.cursor()
    cur.execute("SELECT product_id FROM product LIMIT %d OFFSET %d" % (limit, offset))
 
    rows = cur.fetchall()
    rows = [r[0] for r in list(rows)]
    cur.close()
    return rows


def random_sample(conn, num_of_samples=1):
    cur = conn.cursor()

    products = []
    cur.execute("SELECT product_id FROM product ORDER BY RANDOM() LIMIT %d;" % num_of_samples)

    products += cur.fetchall()
    cur.close()
    return [x[0] for x in products]


def insert_blob(conn, params):
    insert_query = """
        REPLACE INTO 'products'
        (
            'product_id', 'product_unigram_indices', 'product_bigram_indices', 'product_char_trigram_indices',
            'brand_unigram_indices', 'brand_bigram_indices', 'brand_char_trigram_indices',
            'author_unigram_indices', 'author_bigram_indices', 'author_char_trigram_indices',
            'cat_tokens', 'cat_in_product', 'cat_unigram_indices', 'cat_bigram_indices', 'cat_char_trigram_indices', 
            'attr_tokens', 'attr_in_product', 'attr_unigram_indices', 'attr_bigram_indices', 'attr_char_trigram_indices', 
            'features'
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
    """

    cur = conn.cursor()
    cur.execute(insert_query, params)
    conn.commit()
    cur.close()


def get_blob(conn, product_id):
    cur = conn.cursor()
    cur.execute("SELECT * FROM products WHERE product_id='%s'" % product_id)
 
    rows = cur.fetchall()
    rows = list(rows)
    cur.close()
    if len(rows):
        return rows[0]
    return None


def merge(subs, main):
    main_conn = create_connection(main)
    for s in subs:
        subs_conn = create_connection(s) 
        cur = subs_conn.cursor()
        cur.execute("SELECT * FROM products")
        for r in cur:
            insert_blob(main_conn, r)
        
        subs_conn.close()
    main_conn.close()


if __name__ == "__main__":

    conn = create_connection('../db/precomputed-products.db')
    z = get_product(conn, "10601562", table_name="products")
    print(z)
    # for _ in range(50):
    #     print(random_sample(conn, 3))
    conn.close()

    # conn = create_connection("../db/precomputed-products.db")
    # pids = get_batch_product_ids(conn, 31,)