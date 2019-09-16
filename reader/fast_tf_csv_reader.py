import csv
import random
import multiprocessing.queues
from multiprocessing import Process, Manager
from multiprocessing.managers import BaseManager, NamespaceProxy
import time
import numpy as np
import pyhash
import inspect
import json
from vn_lang import query_preprocessing
from reader.convert import convert_strings, convert_cats, convert_attrs, convert_features


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

        self.product_dict = {}
        with open(product_db, "r") as fobj:
            csv_reader= csv.DictReader(fobj)
            for i, r in enumerate(csv_reader):
                r = dict(r)
                r["name"] = query_preprocessing(r.get("name"))
                r["brand"] = query_preprocessing(r.get("brand"))
                r["author"] = " ".join([query_preprocessing(z) for z in r.get("author")])
                self.product_dict[r.get("product_id")] = r
                # if i % 100000 == 0:
                #     print("Loaded %d products" % i)

        self.product_ids =  list(self.product_dict.keys())

    def get(self, pid):
        return self.product_dict.get(pid)

    def sample(self, n):
        return random.sample(self.product_ids, n*7)


def worker(wid, paths, 
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
    
    print("Loading metadata worker %d" % wid)
    meta_inst = MetaData(
            precomputed_path, product_db, vocab_path, 
            cat_tokens_path, attr_tokens_path, maximums_query,
            maximums_product_name, maximums_brand, maximums_author,
            maximums_cat, maximums_attr, unknown_bin)
    print("Metadata loaded worker %d!" % wid)

    total_sample = 0
    while True:
        if queue.qsize() > 10000:
            time.sleep(0.5)
        
        for _ in range(50):
            queries = []
            labels = []
            products = []
            qids = []
            count_keyword = 0

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
                        # neg = meta_inst.sample(n*7)
                        pass
                    else:
                        continue
                else:
                    zero = random.sample(zero, min(len(zero), n*6))
                    # neg = meta_inst.sample(n*7)
                
                for samples, l in zip([pos, zero, neg], [2,1,0]):
                    for s in samples:
                        product = meta_inst.get(s)
                        # if product:
                        #     queries.append(keyword)
                        #     qids.append(count_keyword)
                        #     products.append(product)
                        #     labels.append(l)
                        queries.append(keyword)
                        labels.append(l)

            # product_names = list(map(lambda x: x.get("name"), products))
            # brands = list(map(lambda x: x.get("brand"), products))
            # authors = list(map(lambda x: x.get("author"), products))
            # categories = list(map(lambda x: x.get('categories'), products))
            # attributes = list(map(lambda x: x.get('attributes'), products))
            # features = list(map(lambda x: [x.get(h) for h in meta_inst.precomputed], products))
            # precomputed_features_min = [meta_inst.precomputed.get(h)[0] for h in meta_inst.precomputed]
            # precomputed_features_max = [meta_inst.precomputed.get(h)[1] for h in meta_inst.precomputed]

            # query_unigram_indices, query_bigram_indices, query_char_trigram_indices =  \
            #     convert_strings(
            #         queries, meta_inst.token_2_idx, meta_inst.zero_idx, 
            #         meta_inst.maximums_query[0], meta_inst.maximums_query[1], meta_inst.maximums_query[2], 
            #         unknown_to_idx())
            
            # product_unigram_indices, product_bigram_indices, product_char_trigram_indices =  \
            #     convert_strings(
            #         product_names, meta_inst.token_2_idx, meta_inst.zero_idx, 
            #         meta_inst.maximums_product_name[0], meta_inst.maximums_product_name[1], meta_inst.maximums_product_name[2], 
            #         unknown_to_idx())

            # brand_unigram_indices, brand_bigram_indices, brand_char_trigram_indices =  \
            #     convert_strings(
            #         brands, meta_inst.token_2_idx, meta_inst.zero_idx, 
            #         meta_inst.maximums_brand[0], meta_inst.maximums_brand[1], meta_inst.maximums_brand[2], 
            #         unknown_to_idx())

            # author_unigram_indices, author_bigram_indices, author_char_trigram_indices =  \
            #     convert_strings(
            #         authors, meta_inst.token_2_idx, meta_inst.zero_idx, 
            #         meta_inst.maximums_author[0], meta_inst.maximums_author[1], meta_inst.maximums_author[2], 
            #         unknown_to_idx())

            # cat_tokens, cat_in_product, cat_unigram_indices, cat_bigram_indices, cat_char_trigram_indices = \
            #     convert_cats(
            #         categories,
            #         meta_inst.token_2_idx,
            #         meta_inst.cat_token_2_idx,
            #         meta_inst.zero_idx,
            #         meta_inst.cat_zero_idx,
            #         unknown_to_idx(),
            #         meta_inst.maximums_cat[0], meta_inst.maximums_cat[1], meta_inst.maximums_cat[2]
            #     )

            # attr_tokens, attr_in_product, attr_unigram_indices, attr_bigram_indices, attr_char_trigram_indices = \
            #     convert_attrs(
            #         attributes,
            #         meta_inst.token_2_idx,
            #         meta_inst.attr_token_2_idx,
            #         meta_inst.zero_idx,
            #         meta_inst.attr_zero_idx,
            #         unknown_to_idx(),
            #         meta_inst.maximums_attr[0], meta_inst.maximums_attr[1], meta_inst.maximums_attr[2]
            #     )

            # features = convert_features(
            #     features, precomputed_features_min, precomputed_features_max)

            # labels = np.asarray(labels, dtype=np.int32)
            # qids = np.asarray(qids, dtype=np.int32)

            # print(labels)
            # queue.put([
            #     query_unigram_indices, query_bigram_indices, query_char_trigram_indices, 
            #    product_unigram_indices, product_bigram_indices, product_char_trigram_indices, 
            #    brand_unigram_indices, brand_bigram_indices, brand_char_trigram_indices, 
            #    author_unigram_indices, author_bigram_indices, author_char_trigram_indices, 
            #    cat_tokens, cat_in_product, cat_unigram_indices, cat_bigram_indices, cat_char_trigram_indices,
            #    attr_tokens, attr_in_product, attr_unigram_indices, attr_bigram_indices, attr_char_trigram_indices,
            #    features, count_keyword, qids, labels
            # ])
            queue.put([product, queries, labels])
            total_sample += 1
        if total_sample > limit_sample:
            queue.put(None)
            break


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
        n_workers=8, limit_sample=1000, batch_size=10):
        
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
            ))
            self.procs.append(p)
        
        for p in self.procs:
            p.start()

        self.consumer = self.create_consumer(self.queue)()

    def create_consumer(self, queue):
        def consumer():
            while True:
                result = queue.get()
                if result is None:
                    return
                else:
                    yield tuple(result)
        return consumer
    
    def get_batch(self):
        z = next(self.consumer)
        return z

if __name__ == "__main__":
    import os
    base_dir = "/Users/asm/semranker/transform_impressions"
    paths = [os.path.join(base_dir, f) for f in os.listdir(base_dir)]

    m = ProducerManager(paths, 
        "/Users/asm/semranker/meta/precomputed.json",
        "/Users/asm/semranker/data/product.csv",
        "/Users/asm/semranker/meta/vocab.txt",
        "/Users/asm/semranker/meta/cats.txt",
        "/Users/asm/semranker/meta/attrs.txt")

    r = m.get_batch()
    print(r)
    m.get_batch()