import csv
import random
import multiprocessing.queues
from multiprocessing import Process, Manager
import time
import numpy as np
import pyhash
import inspect
import json
import tensorflow as tf
from vn_lang import query_preprocessing
from reader.convert import convert_strings, convert_cats, convert_attrs, convert_features
from reader.sqlite_product import create_connection, get_product, get_fields, \
    random_sample, get_all_product_ids


class SharedCounter(object):
    """ A synchronized shared counter.
    The locking done by multiprocessing.Value ensures that only a single
    process or thread may read or write the in-memory ctypes object. However,
    in order to do n += 1, Python performs a read followed by a write, so a
    second process may read the old value before the new one is written by the
    first process. The solution is to use a multiprocessing.Lock to guarantee
    the atomicity of the modifications to Value.
    This class comes almost entirely from Eli Bendersky's blog:
    http://eli.thegreenplace.net/2012/01/04/shared-counter-with-pythons-multiprocessing/
    """

    def __init__(self, n = 0):
        self.count = multiprocessing.Value('i', n)

    def increment(self, n = 1):
        """ Increment the counter by n (default = 1) """
        with self.count.get_lock():
            self.count.value += n

    @property
    def value(self):
        """ Return the value of the counter """
        return self.count.value




class PatchQueue(multiprocessing.queues.Queue):
    """ A portable implementation of multiprocessing.Queue.
    Because of multithreading / multiprocessing semantics, Queue.qsize() may
    raise the NotImplementedError exception on Unix platforms like Mac OS X
    where sem_getvalue() is not implemented. This subclass addresses this
    problem by using a synchronized shared counter (initialized to zero) and
    increasing / decreasing its value every time the put() and get() methods
    are called, respectively. This not only prevents NotImplementedError from
    being raised, but also allows us to implement a reliable version of both
    qsize() and empty().
    """

    def __init__(self, *args, **kwargs):
        super(PatchQueue, self).__init__(*args, **kwargs)
        self.size = SharedCounter(0)

    def put(self, *args, **kwargs):
        self.size.increment(1)
        super(PatchQueue, self).put(*args, **kwargs)

    def get(self, *args, **kwargs):
        self.size.increment(-1)
        return super(PatchQueue, self).get(*args, **kwargs)

    def qsize(self):
        """ Reliable implementation of multiprocessing.Queue.qsize() """
        return self.size.value

    def empty(self):
        """ Reliable implementation of multiprocessing.Queue.empty() """
        return not self.qsize()

    def clear(self):
        """ Remove all elements from the Queue. """
        while not self.empty():
            self.get()
            

class MetaData:
    def __init__(self, 
        precomputed_path,
        product_db,
        vocab_path,
        cat_tokens_path, 
        attr_tokens_path,
        maximums_query=[25, 25, 125],#for unigram, bigram, character trigrams
        maximums_product_name=[50, 50, 250], #for unigram, bigram, character trigrams
        maximums_brand=[10, 10, 50],
        maximums_author=[10, 10, 50],
        maximums_cat=[10, 10, 20], #for unigram, bigram, character trigrams
        maximums_attr=[10, 10, 20], #for unigram, bigram, character trigrams
        unknown_bin=8012):

        self.vocab = []
        with open(vocab_path, 'r') as fobj:
            for l in fobj:
                if len(l.strip()):
                    self.vocab.append(l.strip())
        self.cat_tokens = []
        with open(cat_tokens_path, 'r') as fobj:
            for l in fobj:
                if len(l.strip()):
                    self.cat_tokens.append(l.strip())
        self.attr_tokens = []
        with open(attr_tokens_path, 'r') as fobj:
            for l in fobj:
                if len(l.strip()):
                    self.attr_tokens.append(l.strip())

        with open(precomputed_path, 'r') as fobj:
            self.precomputed = json.load(fobj)

        self.vocab_size = len(self.vocab)
        self.cat_tokens_size = len(self.cat_tokens)
        self.attr_tokens_size = len(self.attr_tokens)

        self.unknown_bin = unknown_bin

        self.maximums_query = maximums_query
        self.maximums_product_name = maximums_product_name
        self.maximums_brand = maximums_brand
        self.maximums_author = maximums_author
        self.maximums_cat = maximums_cat
        self.maximums_attr = maximums_attr
        
        self.token_2_idx = {}
        self.cat_token_2_idx = {}
        self.attr_token_2_idx = {}
        
        self.zero_idx = len(self.vocab) + self.unknown_bin
        for i, w in enumerate(self.vocab):
            self.token_2_idx[w] = i
        
        self.cat_zero_idx = len(self.cat_tokens)
        for i, w in enumerate(self.cat_tokens):
            self.cat_token_2_idx[w] = i
        
        self.attr_zero_idx = len(self.attr_tokens)
        for i, w in enumerate(self.attr_tokens):
            self.attr_token_2_idx[w] = i
        
        self.conn = create_connection(product_db)
        self.headers = get_fields(self.conn, table_name="products")

        self.product_ids = get_all_product_ids(self.conn, table_name="products")
    
    def get_product(self, product_id):
        product = get_product(self.conn, product_id, table_name="products")
        if product:
            ret = dict(zip(self.headers, product))
            return ret
        return None


def worker(wid,
    paths, 
    queue, limit_sample, batch_size,
    precomputed_path,
    product_db,
    vocab_path,
    cat_tokens_path,
    attr_tokens_path,
    maximums_query,
    maximums_product_name,
    maximums_brand,
    maximums_author,
    maximums_cat,
    maximums_attr,
    unknown_bin):
    fobjs = []
    readers = []
    hasher = pyhash.murmur3_32()
    
    def unknown_to_idx():
        def _inside(unknown):
            return hasher(unknown) % unknown_bin
        return _inside
    for p in paths:
        fobj = open(p, "r")
        reader = csv.reader(fobj)
        fobjs.append(fobj)
        readers.append(reader)
    
    meta_inst = MetaData(
            precomputed_path, product_db, vocab_path, 
            cat_tokens_path, attr_tokens_path, maximums_query,
            maximums_product_name, maximums_brand, maximums_author,
            maximums_cat, maximums_attr, unknown_bin)
    product_ids = meta_inst.product_ids
    print("Data worker %d started" % wid)

    total_sample = 0
    while True:
        if queue.qsize() > 1000:
            time.sleep(0.2)
        
        for _ in range(50):
            queries = []
            labels = []
            products = []
            qids = []
            count_keyword = 0
            unique_queries = []
            count_qs = []
            count_t = 0
            for k in range(batch_size):
                idx = k % len(readers)
                reader = readers[idx]
                r = next(reader)
                if r is None:
                    # reset reader
                    fobj = fobjs[idx].seek(0)
                    r = next(reader)
                keyword = r[0]
                r1 = r[1]
                if len(keyword) == 0:
                    continue
                count_keyword += 1
                pk = r1.split("|")
                pnk = [z.split("#") for z in pk]
                pos = []
                zero = []
                neg = []

                for p in pnk:
                    if p[1] == '2':
                        pos.append(p[0])
                    elif p[1] == '1':
                        zero.append(p[0])
                    else:
                        neg.append(p[0])
                n = len(pos)
                if n > 6:
                    n = 4
                    pos = random.sample(pos, n)
                if n == 0: 
                    n = len(zero)
                    if n > 6:
                        n = 4
                        zero = random.sample(zero, n)
                    if n:
                        neg = random.sample(product_ids, n*7)
                        pass
                    else:
                        continue
                else:
                    zero = random.sample(zero, min(len(zero), n*6))
                    neg = random.sample(product_ids, n*7)
                
                count_q = 0
                for samples, l in zip([pos, zero, neg], [2,1,0]):
                    for s in samples:
                        product = meta_inst.get_product(s)
                        if product:
                            count_q += 1
                            queries.append(keyword)
                            qids.append(count_keyword)
                            products.append(product)
                            labels.append(l)
                if count_q:
                    unique_queries.append(keyword)
                    count_qs.append(count_q)

            query_unigram_indices = []
            query_bigram_indices = []
            query_char_trigram_indices = []

            for q, r in zip(unique_queries, count_qs):
                u, b, t =  \
                convert_strings(
                    [q], meta_inst.token_2_idx, meta_inst.zero_idx, 
                    meta_inst.maximums_query[0], meta_inst.maximums_query[1], meta_inst.maximums_query[2], 
                    unknown_to_idx())
                query_unigram_indices.append(np.tile(u, (r, 1)))
                query_bigram_indices.append(np.tile(b, (r, 1)))
                query_char_trigram_indices.append(np.tile(t, (r, 1)))
            query_unigram_indices = np.concatenate(query_unigram_indices, axis=0)
            query_bigram_indices = np.concatenate(query_bigram_indices, axis=0)
            query_char_trigram_indices = np.concatenate(query_char_trigram_indices, axis=0)

            product_unigram_indices = []
            product_bigram_indices = []
            product_char_trigram_indices = []

            brand_unigram_indices = []
            brand_bigram_indices = []
            brand_char_trigram_indices = []

            author_unigram_indices = []
            author_bigram_indices = []
            author_char_trigram_indices = []

            cat_tokens = []
            cat_in_product = []
            cat_unigram_indices = []
            cat_bigram_indices = []
            cat_char_trigram_indices = []

            attr_tokens = []
            attr_in_product = [] 
            attr_unigram_indices = [] 
            attr_bigram_indices = [] 
            attr_char_trigram_indices = []

            features = []

            for p in products:
                product_unigram_indices.append(
                    np.frombuffer(p.get("product_unigram_indices"), dtype=np.int32))
                product_bigram_indices.append(
                    np.frombuffer(p.get("product_bigram_indices"), dtype=np.int32))
                product_char_trigram_indices.append(
                    np.frombuffer(p.get("product_char_trigram_indices"), dtype=np.int32))
                brand_unigram_indices.append(
                    np.frombuffer(p.get("brand_unigram_indices"), dtype=np.int32))
                brand_bigram_indices.append(
                    np.frombuffer(p.get("brand_bigram_indices"), dtype=np.int32))
                brand_char_trigram_indices.append(
                    np.frombuffer(p.get("brand_char_trigram_indices"), dtype=np.int32))
                author_unigram_indices.append(
                    np.frombuffer(p.get("author_unigram_indices"), dtype=np.int32))
                author_bigram_indices.append(
                    np.frombuffer(p.get("author_bigram_indices"), dtype=np.int32))
                author_char_trigram_indices.append(
                    np.frombuffer(p.get("author_char_trigram_indices"), dtype=np.int32))

                cat_tokens.append(
                    np.frombuffer(p.get("cat_tokens"), dtype=np.int32)
                )
                cip = int(np.frombuffer(p.get("cat_in_product"), dtype=np.int32))
                cat_in_product.append(cip)
                cat_unigram_indices.append(
                    np.reshape(np.frombuffer(p.get("cat_unigram_indices"), dtype=np.int32), 
                    (cip, meta_inst.maximums_cat[0]))
                )
                cat_bigram_indices.append(
                    np.reshape(np.frombuffer(p.get("cat_bigram_indices"), dtype=np.int32), 
                    (cip, meta_inst.maximums_cat[1]))
                )
                cat_char_trigram_indices.append(
                    np.reshape(np.frombuffer(p.get("cat_char_trigram_indices"), dtype=np.int32), 
                    (cip, meta_inst.maximums_cat[2]))
                )

                attr_tokens.append(
                    np.frombuffer(p.get("attr_tokens"), dtype=np.int32)
                )
                aip = int(np.frombuffer(p.get("attr_in_product"), dtype=np.int32))
                attr_in_product.append(aip)
                attr_unigram_indices.append(
                   np.reshape(np.frombuffer(p.get("attr_unigram_indices"), dtype=np.int32), 
                    (aip, meta_inst.maximums_attr[0])) 
                )
                attr_bigram_indices.append(
                   np.reshape(np.frombuffer(p.get("attr_bigram_indices"), dtype=np.int32), 
                    (aip, meta_inst.maximums_attr[1])) 
                )
                attr_char_trigram_indices.append(
                   np.reshape(np.frombuffer(p.get("attr_char_trigram_indices"), dtype=np.int32), 
                    (aip, meta_inst.maximums_attr[2])) 
                )

                features.append(
                    np.frombuffer(p.get("features"), dtype=np.float32)
                )

            product_unigram_indices = np.stack(product_unigram_indices)
            product_bigram_indices = np.stack(product_bigram_indices)
            product_char_trigram_indices = np.stack(product_char_trigram_indices)

            brand_unigram_indices = np.stack(brand_unigram_indices)
            brand_bigram_indices = np.stack(brand_bigram_indices)
            brand_char_trigram_indices = np.stack(brand_char_trigram_indices)

            author_unigram_indices = np.stack(author_unigram_indices)
            author_bigram_indices = np.stack(author_bigram_indices)
            author_char_trigram_indices = np.stack(author_char_trigram_indices)

            cat_tokens = np.concatenate(cat_tokens)
            cat_in_product = np.array(cat_in_product, dtype=np.int32)
            cat_unigram_indices = np.concatenate(cat_unigram_indices, axis=0)
            cat_bigram_indices = np.concatenate(cat_bigram_indices, axis=0)
            cat_char_trigram_indices = np.concatenate(cat_char_trigram_indices, axis=0)
            
            attr_tokens = np.concatenate(attr_tokens)
            attr_in_product = np.array(attr_in_product, dtype=np.int32)
            attr_unigram_indices = np.concatenate(attr_unigram_indices, axis=0)
            attr_bigram_indices = np.concatenate(attr_bigram_indices, axis=0)
            attr_char_trigram_indices = np.concatenate(attr_char_trigram_indices, axis=0)

            features = np.stack(features)

            labels = np.asarray(labels, dtype=np.int32)
            qids = np.asarray(qids, dtype=np.int32)

            queue.put([
               query_unigram_indices, query_bigram_indices, query_char_trigram_indices, 
               product_unigram_indices, product_bigram_indices, product_char_trigram_indices, 
               brand_unigram_indices, brand_bigram_indices, brand_char_trigram_indices, 
               author_unigram_indices, author_bigram_indices, author_char_trigram_indices, 
               cat_tokens, cat_in_product, cat_unigram_indices, cat_bigram_indices, cat_char_trigram_indices,
               attr_tokens, attr_in_product, attr_unigram_indices, attr_bigram_indices, attr_char_trigram_indices,
               features, count_keyword, qids, labels
            ])

            total_sample += 1
            if total_sample > limit_sample:
                queue.put(None)
                break
            
        if total_sample > limit_sample:
            # queue.put(None)
            break
    meta_inst.conn.close()
    print("Worker-%d Exiting" % wid)



class ProducerManager:
    def __init__(self, paths, 
        precomputed_path,
        product_db,
        vocab_path,
        cat_tokens_path, 
        attr_tokens_path,
        maximums_query=[25, 25, 125],#for unigram, bigram, character trigrams
        maximums_product_name=[50, 50, 250], #for unigram, bigram, character trigrams
        maximums_brand=[10, 10, 50],
        maximums_author=[10, 10, 50],
        maximums_cat=[10, 10, 20], #for unigram, bigram, character trigrams
        maximums_attr=[10, 10, 20], #for unigram, bigram, character trigrams
        unknown_bin=8012, 
        n_workers=1, limit_sample=10, batch_size=2, warmup=60):


        self.queue = PatchQueue(ctx=multiprocessing.get_context())
        paths = sorted(paths)
        self.procs = []
        c = int(len(paths) / n_workers)

        for i in range(n_workers):
            sub_paths = paths[i*c:(i+1)*c]
            p = Process(target=worker, args=(i,
                sub_paths, self.queue, limit_sample, 
                batch_size,
                precomputed_path,
                product_db,
                vocab_path,
                cat_tokens_path,
                attr_tokens_path,
                maximums_query,
                maximums_product_name,
                maximums_brand,
                maximums_author,
                maximums_cat,
                maximums_attr,
                unknown_bin
            ),daemon=True)
            self.procs.append(p)
        
        for p in self.procs:
            p.start()

        self.maximums_query = maximums_query
        self.maximums_product_name = maximums_product_name
        self.maximums_brand = maximums_brand
        self.maximums_author = maximums_author
        self.maximums_cat = maximums_cat
        self.maximums_attr = maximums_attr

        self.vocab = []
        with open(vocab_path, 'r') as fobj:
            for l in fobj:
                if len(l.strip()):
                    self.vocab.append(l.strip())
        self.cat_tokens = []
        with open(cat_tokens_path, 'r') as fobj:
            for l in fobj:
                if len(l.strip()):
                    self.cat_tokens.append(l.strip())
        self.attr_tokens = []
        with open(attr_tokens_path, 'r') as fobj:
            for l in fobj:
                if len(l.strip()):
                    self.attr_tokens.append(l.strip())

        with open(precomputed_path, 'r') as fobj:
            self.precomputed = json.load(fobj)

        self.vocab_size = len(self.vocab)
        self.cat_tokens_size = len(self.cat_tokens)
        self.attr_tokens_size = len(self.attr_tokens)

        self.unknown_bin = unknown_bin


        self.consumer = self.create_consumer(self.queue)
        if warmup > 0:
            print("Warmup %d seconds" % warmup)
            time.sleep(warmup)

    def create_consumer(self, queue):
        def consumer():
            while True:
                result = queue.get()
                if result is None:
                    return
                else:
                    yield tuple(result)
        return consumer
    
    def tf_map(self):
        def _inside(query_unigram_indices, query_bigram_indices, query_char_trigram_indices, 
               product_unigram_indices, product_bigram_indices, product_char_trigram_indices, 
               brand_unigram_indices, brand_bigram_indices, brand_char_trigram_indices, 
               author_unigram_indices, author_bigram_indices, author_char_trigram_indices, 
               cat_tokens, cat_in_product, cat_unigram_indices, cat_bigram_indices, cat_char_trigram_indices,
               attr_tokens, attr_in_product, attr_unigram_indices, attr_bigram_indices, attr_char_trigram_indices,
               features, number_of_queries, qids, labels):
            
            return {
                'query_unigram_indices' :tf.squeeze(query_unigram_indices, axis=0), 
                'query_bigram_indices': tf.squeeze(query_bigram_indices, axis=0),
                'query_char_trigram_indices': tf.squeeze(query_char_trigram_indices, axis=0),
                'product_unigram_indices': tf.squeeze(product_unigram_indices, axis=0),
                'product_bigram_indices': tf.squeeze(product_bigram_indices, axis=0),
                'product_char_trigram_indices': tf.squeeze(product_char_trigram_indices, axis=0),
                'brand_unigram_indices': tf.squeeze(brand_unigram_indices, axis=0),
                'brand_bigram_indices': tf.squeeze(brand_bigram_indices, axis=0),
                'brand_char_trigram_indices': tf.squeeze(brand_char_trigram_indices, axis=0),
                'author_unigram_indices': tf.squeeze(author_unigram_indices, axis=0),
                'author_bigram_indices': tf.squeeze(author_bigram_indices, axis=0),
                'author_char_trigram_indices': tf.squeeze(author_char_trigram_indices, axis=0),
                'cat_tokens': tf.squeeze(cat_tokens, axis=0),
                'cat_in_product': tf.squeeze(cat_in_product, axis=0),
                'cat_unigram_indices': tf.squeeze(cat_unigram_indices, axis=0),
                'cat_bigram_indices': tf.squeeze(cat_bigram_indices, axis=0),
                'cat_char_trigram_indices': tf.squeeze(cat_char_trigram_indices, axis=0),
                'attr_tokens': tf.squeeze(attr_tokens, axis=0),
                'attr_in_product': tf.squeeze(attr_in_product, axis=0),
                'attr_unigram_indices': tf.squeeze(attr_unigram_indices, axis=0),
                'attr_bigram_indices': tf.squeeze(attr_bigram_indices, axis=0),
                'attr_char_trigram_indices': tf.squeeze(attr_char_trigram_indices, axis=0),
                'features':tf.squeeze(features, axis=0),
                'number_of_queries': tf.squeeze(number_of_queries, axis=0),
                'qids': tf.squeeze(qids, axis=0)}, {"labels": tf.squeeze(labels, axis=0)}
        return _inside

    def get_batch(self):
        dataset = tf.data.Dataset.from_generator(self.consumer,
            output_types=(
                tf.int32, tf.int32, tf.int32, # query_unigram_indices, query_bigram_indices, query_char_trigram_indices, 
                tf.int32, tf.int32, tf.int32, # product_unigram_indices, product_bigram_indices, product_char_trigram_indices
                tf.int32, tf.int32, tf.int32, # brand_unigram_indices, brand_bigram_indices, brand_char_trigram_indices
                tf.int32, tf.int32, tf.int32, # author_unigram_indices, author_bigram_indices, author_char_trigram_indices
                tf.int32, tf.int32, tf.int32, tf.int32, tf.int32, # cat_tokens, cat_in_product, cat_unigram_indices, cat_bigram_indices, cat_char_trigram_indices
                tf.int32, tf.int32, tf.int32, tf.int32, tf.int32, # attr_tokens, attr_in_product, attr_unigram_indices, attr_bigram_indices, attr_char_trigram_indices
                tf.float32, tf.int32, tf.int32, tf.int32 # features, number_of_queries, qids, labels
            ),
            output_shapes=(
                tf.TensorShape([None, self.maximums_query[0]]), tf.TensorShape([None, self.maximums_query[1]]), tf.TensorShape([None, self.maximums_query[2]]),
                tf.TensorShape([None, self.maximums_product_name[0]]), tf.TensorShape([None, self.maximums_product_name[1]]), tf.TensorShape([None, self.maximums_product_name[2]]),
                tf.TensorShape([None, self.maximums_brand[0]]), tf.TensorShape([None, self.maximums_brand[1]]), tf.TensorShape([None, self.maximums_brand[2]]),
                tf.TensorShape([None, self.maximums_author[0]]), tf.TensorShape([None, self.maximums_author[1]]), tf.TensorShape([None, self.maximums_author[2]]),
                tf.TensorShape([None]), tf.TensorShape([None]), tf.TensorShape([None, self.maximums_cat[0]]), tf.TensorShape([None, self.maximums_cat[1]]), tf.TensorShape([None, self.maximums_cat[2]]),
                tf.TensorShape([None]), tf.TensorShape([None]), tf.TensorShape([None, self.maximums_attr[0]]), tf.TensorShape([None, self.maximums_attr[1]]), tf.TensorShape([None, self.maximums_attr[2]]),
                tf.TensorShape([None, len(self.precomputed)]), tf.TensorShape([]), tf.TensorShape([None]), tf.TensorShape([None])
            ))
        batch = dataset.batch(1)
        batch = batch.repeat(None)
        batch = batch.map(self.tf_map())

        iterator = batch.make_one_shot_iterator()
        next_element = iterator.get_next()

        return next_element
    
    def input_fn_generator(self):
        return lambda: self.get_batch()
        


if __name__ == "__main__":
    import os
    base_dir = "/home/asm/semranker/overfit"
    paths = [os.path.join(base_dir, f) for f in os.listdir(base_dir)]

    # m = ProducerManager(paths, 
    #     "/home/asm/semranker/meta/precomputed.json",
    #     "/home/asm/semranker/db/precomputed-products.db",
    #     "/home/asm/semranker/meta/vocab.txt",
    #     "/home/asm/semranker/meta/cats.txt",
    #     "/home/asm/semranker/meta/attrs.txt")

    

    # begin_ = time.time()
    # #for _ in range(6):
    # m.get_batch()
    # end_ = time.time()

    # print("Get 100 batch in:%0.4f" % (end_ - begin_))

    m = ProducerManager(paths, 
        "/home/asm/semranker/meta/precomputed.json",
        "/home/asm/semranker/db/precomputed-products.db",
        "/home/asm/semranker/meta/vocab.txt",
        "/home/asm/semranker/meta/cats.txt",
        "/home/asm/semranker/meta/attrs.txt",
        n_workers=1, limit_sample=10, batch_size=2, warmup=10)

    batch = m.get_batch()
    init_op = tf.initializers.global_variables()

    with tf.Session() as sess:
        sess.run(init_op)
        begin_ = time.time()
        for _ in range(1):
            v = sess.run(batch)
            print(v)
        end_ = time.time()
        print("Get 10 batch in:%0.4f" % (end_ - begin_))

        