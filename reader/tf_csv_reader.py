import tensorflow as tf
import pyhash
import csv
import json
import random
from random import randint
import numpy as np
from reader.sqlite_product import create_connection, get_product, get_fields, random_sample
from reader.convert import convert_strings, convert_cats, convert_attrs, convert_features
from vn_lang import query_preprocessing
import time

class CsvSemRankerReader(object):
    def __init__(
        self, pair_paths,
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

        self.hasher = pyhash.murmur3_32()

        # initialize sampling pools
        self.pair_paths = pair_paths
        self.precomputed_path = precomputed_path

        # self.conn = create_connection(product_db)
        # self.headers = get_fields(self.conn)

        if product_db:
            self.product_dict = {}
            with open(product_db, "r") as fobj:
                csv_reader= csv.DictReader(fobj)
                for i, r in enumerate(csv_reader):
                    r = dict(r)
                    r["name"] = query_preprocessing(r.get("name"))
                    r["brand"] = query_preprocessing(r.get("brand"))
                    r["author"] = " ".join([query_preprocessing(z) for z in r.get("author")])
                    self.product_dict[r.get("product_id")] = r
                    if i % 100000 == 0:
                        print("Loaded %d products" % i)

            self.product_ids =  list(self.product_dict.keys())

    def unknown_to_idx(self, unknown):
        return self.hasher(unknown) % self.unknown_bin
    
    def get_product(self, product_id):
        product = get_product(self.conn, product_id)
        if product:
            ret = dict(zip(self.headers, product))
            if len(ret['name']):
                for k in ret:
                    if k in self.precomputed:
                        if len(ret[k]) == 0:
                            ret[k] = 0.
                        else:
                            ret[k] = float(ret[k])
            else:
                return None
            return ret
        return None

    def _wrapper_map(self):
        def _map_to_indices(keywords, interactions):
            queries = []
            labels = []
            products = []
            keywords = keywords.numpy()
            interactions = interactions.numpy()
            qids = []
            count_keyword = 0

            for pairs in zip(keywords, interactions):
                dd = time.time()
                keyword = pairs[0].decode()
                if len(keyword) == 0:
                    continue
                count_keyword += 1
                r1 = pairs[1].decode()
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
                        # neg = random.sample(neg, min(len(neg), int(n*12)))
                        neg = random.sample(self.product_ids, n*10)
                    else:
                        continue
                else:
                    zero = random.sample(zero, min(len(zero), n*6))
                    # neg = random.sample(self.product_ids, n*2) + random.sample(neg, min(len(neg), n*10))
                    neg = random.sample(self.product_ids, n*10)

                for samples, l in zip([pos, zero, neg], [2,1,0]):
                    for s in samples:
                        product = self.product_dict.get(s)
                        if product:
                            queries.append(keyword)
                            qids.append(count_keyword)
                            products.append(product)
                            labels.append(l)

            product_names = list(map(lambda x: x.get("name"), products))
            brands = list(map(lambda x: x.get("brand"), products))
            authors = list(map(lambda x: x.get("author"), products))
            categories = list(map(lambda x: x.get('categories'), products))
            attributes = list(map(lambda x: x.get('attributes'), products))
            features = list(map(lambda x: [x.get(h) for h in self.precomputed], products))
            precomputed_features_min = [self.precomputed.get(h)[0] for h in self.precomputed]
            precomputed_features_max = [self.precomputed.get(h)[1] for h in self.precomputed]

            query_unigram_indices, query_bigram_indices, query_char_trigram_indices =  \
                convert_strings(
                    queries, self.token_2_idx, self.zero_idx, 
                    self.maximums_query[0], self.maximums_query[1], self.maximums_query[2], 
                    self.unknown_to_idx)
            
            product_unigram_indices, product_bigram_indices, product_char_trigram_indices =  \
                convert_strings(
                    product_names, self.token_2_idx, self.zero_idx, 
                    self.maximums_product_name[0], self.maximums_product_name[1], self.maximums_product_name[2], 
                    self.unknown_to_idx)

            brand_unigram_indices, brand_bigram_indices, brand_char_trigram_indices =  \
                convert_strings(
                    brands, self.token_2_idx, self.zero_idx, 
                    self.maximums_brand[0], self.maximums_brand[1], self.maximums_brand[2], 
                    self.unknown_to_idx)

            author_unigram_indices, author_bigram_indices, author_char_trigram_indices =  \
                convert_strings(
                    authors, self.token_2_idx, self.zero_idx, 
                    self.maximums_author[0], self.maximums_author[1], self.maximums_author[2], 
                    self.unknown_to_idx)

            cat_tokens, cat_in_product, cat_unigram_indices, cat_bigram_indices, cat_char_trigram_indices = \
                convert_cats(
                    categories,
                    self.token_2_idx,
                    self.cat_token_2_idx,
                    self.zero_idx,
                    self.cat_zero_idx,
                    self.unknown_to_idx,
                    self.maximums_cat[0], self.maximums_cat[1], self.maximums_cat[2]
                )

            attr_tokens, attr_in_product, attr_unigram_indices, attr_bigram_indices, attr_char_trigram_indices = \
                convert_attrs(
                    attributes,
                    self.token_2_idx,
                    self.attr_token_2_idx,
                    self.zero_idx,
                    self.attr_zero_idx,
                    self.unknown_to_idx,
                    self.maximums_attr[0], self.maximums_attr[1], self.maximums_attr[2]
                )

            features = convert_features(
                features, precomputed_features_min, precomputed_features_max)

            labels = np.asarray(labels, dtype=np.int32)
            qids = np.asarray(qids, dtype=np.int32)
            
            return query_unigram_indices, query_bigram_indices, query_char_trigram_indices, \
               product_unigram_indices, product_bigram_indices, product_char_trigram_indices, \
               brand_unigram_indices, brand_bigram_indices, brand_char_trigram_indices, \
               author_unigram_indices, author_bigram_indices, author_char_trigram_indices, \
               cat_tokens, cat_in_product, cat_unigram_indices, cat_bigram_indices, cat_char_trigram_indices,\
               attr_tokens, attr_in_product, attr_unigram_indices, attr_bigram_indices, attr_char_trigram_indices,\
               features, count_keyword, qids, labels
        return _map_to_indices

    def tf_map(self):
        def _inside(x, y):
            query_unigram_indices, query_bigram_indices, query_char_trigram_indices, \
            product_unigram_indices, product_bigram_indices, product_char_trigram_indices, \
            brand_unigram_indices, brand_bigram_indices, brand_char_trigram_indices, \
            author_unigram_indices, author_bigram_indices, author_char_trigram_indices, \
            cat_tokens, cat_in_product, cat_unigram_indices, cat_bigram_indices, cat_char_trigram_indices, \
            attr_tokens, attr_in_product, attr_unigram_indices, attr_bigram_indices, attr_char_trigram_indices, \
            features, number_of_queries, qids, labels = tf.py_function(
                self._wrapper_map(), [x, y], 
                [
                    tf.int32, tf.int32, tf.int32, # query_unigram_indices, query_bigram_indices, query_char_trigram_indices
                    tf.int32, tf.int32, tf.int32, # product_unigram_indices, product_bigram_indices, product_char_trigram_indices
                    tf.int32, tf.int32, tf.int32, # brand_unigram_indices, brand_bigram_indices, brand_char_trigram_indices
                    tf.int32, tf.int32, tf.int32, # author_unigram_indices, author_bigram_indices, author_char_trigram_indices
                    tf.int32, tf.int32, tf.int32, tf.int32, tf.int32, # cat_tokens, cat_in_product, cat_unigram_indices, cat_bigram_indices, cat_char_trigram_indices
                    tf.int32, tf.int32, tf.int32, tf.int32, tf.int32, # attr_tokens, attr_in_product, attr_unigram_indices, attr_bigram_indices, attr_char_trigram_indices
                    tf.float32, tf.int32, tf.int32, tf.int32 # features, number_of_queries, qids, labels

                ])

            query_unigram_indices = tf.reshape(
                query_unigram_indices, [-1, self.maximums_query[0]])
            query_bigram_indices = tf.reshape(
                query_bigram_indices, [-1, self.maximums_query[1]]) 
            query_char_trigram_indices = tf.reshape(
                query_char_trigram_indices, [-1, self.maximums_query[2]])

            product_unigram_indices = tf.reshape(
                product_unigram_indices, [-1, self.maximums_product_name[0]])
            product_bigram_indices = tf.reshape(
                product_bigram_indices, [-1, self.maximums_product_name[1]])
            product_char_trigram_indices = tf.reshape(
                product_char_trigram_indices, [-1, self.maximums_product_name[2]])

            brand_unigram_indices = tf.reshape(
                brand_unigram_indices, [-1, self.maximums_brand[0]])
            brand_bigram_indices = tf.reshape(
                brand_bigram_indices, [-1, self.maximums_brand[1]])
            brand_char_trigram_indices = tf.reshape(
                brand_char_trigram_indices, [-1, self.maximums_brand[2]])

            author_unigram_indices = tf.reshape(
                author_unigram_indices, [-1, self.maximums_author[0]])
            author_bigram_indices = tf.reshape(
                author_bigram_indices, [-1, self.maximums_author[1]])
            author_char_trigram_indices = tf.reshape(
                author_char_trigram_indices, [-1, self.maximums_author[2]])

            cat_tokens = tf.reshape(cat_tokens, [-1]) 
            cat_in_product = tf.reshape(cat_in_product, [-1])
            cat_unigram_indices = tf.reshape(
                cat_unigram_indices, [-1, self.maximums_cat[0]])
            cat_bigram_indices = tf.reshape(
                cat_bigram_indices, [-1, self.maximums_cat[1]])
            cat_char_trigram_indices = tf.reshape(
                cat_char_trigram_indices, [-1, self.maximums_cat[2]])
            
            attr_tokens = tf.reshape(attr_tokens, [-1]) 
            attr_in_product = tf.reshape(attr_in_product, [-1])
            attr_unigram_indices = tf.reshape(
                attr_unigram_indices, [-1, self.maximums_attr[0]])
            attr_bigram_indices = tf.reshape(
                attr_bigram_indices, [-1, self.maximums_attr[1]])
            attr_char_trigram_indices = tf.reshape(
                attr_char_trigram_indices, [-1, self.maximums_attr[2]])

            features = tf.reshape(features, [-1, len(self.precomputed)])
            labels = tf.reshape(labels, [-1])
            qids = tf.reshape(qids, [-1])
            # weights = tf.reshape(weights, [-1])

            return {
                'query_unigram_indices' :query_unigram_indices, 
                'query_bigram_indices': query_bigram_indices, 
                'query_char_trigram_indices': query_char_trigram_indices,
                'product_unigram_indices':product_unigram_indices, 
                'product_bigram_indices': product_bigram_indices, 
                'product_char_trigram_indices': product_char_trigram_indices, 
                'brand_unigram_indices': brand_unigram_indices, 
                'brand_bigram_indices': brand_bigram_indices, 
                'brand_char_trigram_indices': brand_char_trigram_indices, 
                'author_unigram_indices': author_unigram_indices, 
                'author_bigram_indices': author_bigram_indices, 
                'author_char_trigram_indices': author_char_trigram_indices, 
                'cat_tokens': cat_tokens, 
                'cat_in_product': cat_in_product, 
                'cat_unigram_indices': cat_unigram_indices, 
                'cat_bigram_indices': cat_bigram_indices, 
                'cat_char_trigram_indices': cat_char_trigram_indices, 
                'attr_tokens': attr_tokens, 
                'attr_in_product': attr_in_product, 
                'attr_unigram_indices': attr_unigram_indices, 
                'attr_bigram_indices': attr_bigram_indices,
                'attr_char_trigram_indices': attr_char_trigram_indices, 
                'features':features,
                'number_of_queries': number_of_queries,
                'qids': qids}, {"labels":labels} #{"labels":labels, "weights": weights}
        return _inside


    def get_batch(self, batch_size, for_estimator=False):
        with tf.device('/device:CPU:*'):
            self.pair_data = tf.data.experimental.CsvDataset(
                filenames=self.pair_paths,
                record_defaults=[tf.string, tf.string])
            pair_dataset = self.pair_data.batch(batch_size)
            pair_dataset = pair_dataset.repeat(None)
            pair_dataset = pair_dataset.map(self.tf_map())

            if for_estimator:
                return pair_dataset

            iterator = pair_dataset.make_one_shot_iterator()
            next_element = iterator.get_next()

            return next_element

    def input_fn_generator(self, batch_size):
        return lambda: self.get_batch(batch_size, for_estimator=True)

    def end(self):
        self.conn.close()