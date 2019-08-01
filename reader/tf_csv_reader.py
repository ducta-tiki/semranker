import tensorflow as tf
import pyhash
import csv
import json
from random import randint
import numpy as np
from reader.sqlite_product import create_connection, get_product, get_fields
from reader.convert import convert_strings, convert_cats, convert_attrs, convert_features
from vn_lang import query_preprocessing


class CsvSemRankerReader(object):
    def __init__(
        self, pos_path, zero_path, neg_path,
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
        self.pos_path = pos_path
        self.zero_path = zero_path
        self.neg_path = neg_path
        self.precomputed_path = precomputed_path

        self.pos_pool = []
        self.zero_pool = []
        self.neg_pool = []

        pos_fobj = open(pos_path, 'r')
        pos_reader = csv.reader(pos_fobj)

        zero_fobj = open(zero_path, 'r')
        zero_reader = csv.reader(zero_fobj)

        neg_fobj = open(neg_path, 'r')
        neg_reader = csv.reader(neg_fobj)

        for reader, pool, label in zip(
            [self.pos_pool, self.zero_pool, self.neg_pool], 
            [pos_reader, zero_reader, neg_reader], [2, 1, 0]):
            for r in reader:
                if len(pool) < 10000:
                    pool.insert(randint(0, len(pool)), [r[0], r[1], label])
                else:
                    break
        pos_fobj.close(); zero_fobj.close(); neg_fobj.close()

        self.conn = create_connection(product_db)
        self.headers = get_fields(self.conn)

        self.pos_data = tf.contrib.data.CsvDataset(
            filenames=[self.pos_path],
            record_defaults=[tf.string, tf.string])
        self.neg_data = tf.contrib.data.CsvDataset(
            filenames=[self.neg_path],
            record_defaults=[tf.string, tf.string])
        self.zero_data = tf.contrib.data.CsvDataset(
            filenames=[self.zero_path],
            record_defaults=[tf.string, tf.string])

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
        def _map_to_indices(batch_pos, batch_zero, batch_neg):
            z = lambda obj, label: [[x.decode(), y.decode(), label] for x, y in zip(list(obj[0]), list(obj[1]))]
            pos = batch_pos.numpy(); pos = z(pos, 2)
            zero = batch_zero.numpy(); zero = z(zero, 1)
            neg = batch_neg.numpy(); neg = z(neg, 0)

            new_pos = []
            new_zero = []
            new_neg = []
            for pool, new_batch, new_store in zip(
                [self.pos_pool, self.zero_pool, self.neg_pool], 
                [pos, zero, neg], [new_pos, new_zero, new_neg]):
                for s in new_batch:
                    pool.insert(randint(0, len(pool)), s)
                for i in range(len(pos)):
                    new_store.append(pool.pop())
            products = []
            queries = []
            labels = []
            for sample in pos + zero + neg:
                product = self.get_product(sample[1])
                if product:
                    queries.append(sample[0])
                    products.append(product)
                    labels.append(sample[2])
            product_names = list(map(lambda x: query_preprocessing(x.get("name")), products))
            brands = list(map(lambda x: query_preprocessing(x.get("brand")), products))
            authors = list(map(lambda x: " ".join([query_preprocessing(z) for z in x.get("author")]), products))
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
            return query_unigram_indices, query_bigram_indices, query_char_trigram_indices, \
               product_unigram_indices, product_bigram_indices, product_char_trigram_indices, \
               brand_unigram_indices, brand_bigram_indices, brand_char_trigram_indices, \
               author_unigram_indices, author_bigram_indices, author_char_trigram_indices, \
               cat_tokens, cat_in_product, cat_unigram_indices, cat_bigram_indices, cat_char_trigram_indices,\
               attr_tokens, attr_in_product, attr_unigram_indices, attr_bigram_indices, attr_char_trigram_indices,\
               features, labels
        return _map_to_indices

    def get_batch(self, batch_size):
        with tf.device('/device:CPU:*'):
            pos_dataset = self.pos_data.batch(batch_size)
            pos_dataset = pos_dataset.repeat(None)
            iterator = pos_dataset.make_one_shot_iterator()
            next_pos_batch = iterator.get_next()

            zero_dataset = self.zero_data.batch(batch_size)
            zero_dataset = zero_dataset.repeat(None)
            iterator = zero_dataset.make_one_shot_iterator()
            next_zero_batch = iterator.get_next()

            neg_dataset = self.neg_data.batch(batch_size)
            neg_dataset = neg_dataset.repeat(None)
            iterator = neg_dataset.make_one_shot_iterator()
            next_neg_batch = iterator.get_next()

            next_batch = tf.py_function(
                self._wrapper_map(), [next_pos_batch, next_zero_batch, next_neg_batch], 
                [
                    tf.int32, tf.int32, tf.int32, # query_unigram_indices, query_bigram_indices, query_char_trigram_indices
                    tf.int32, tf.int32, tf.int32, # product_unigram_indices, product_bigram_indices, product_char_trigram_indices
                    tf.int32, tf.int32, tf.int32, # brand_unigram_indices, brand_bigram_indices, brand_char_trigram_indices
                    tf.int32, tf.int32, tf.int32, # author_unigram_indices, author_bigram_indices, author_char_trigram_indices
                    tf.int32, tf.int32, tf.int32, tf.int32, tf.int32, # cat_tokens, cat_in_product, cat_unigram_indices, cat_bigram_indices, cat_char_trigram_indices
                    tf.int32, tf.int32, tf.int32, tf.int32, tf.int32, # attr_tokens, attr_in_product, attr_unigram_indices, attr_bigram_indices, attr_char_trigram_indices
                    tf.float32, tf.int32 # features, labels

                ])

        return next_batch

    def input_fn_generator(self, batch_size):
        def _inside():
            query_unigram_indices, query_bigram_indices, query_char_trigram_indices, \
               product_unigram_indices, product_bigram_indices, product_char_trigram_indices, \
               brand_unigram_indices, brand_bigram_indices, brand_char_trigram_indices, \
               author_unigram_indices, author_bigram_indices, author_char_trigram_indices, \
               cat_tokens, cat_in_product, cat_unigram_indices, cat_bigram_indices, cat_char_trigram_indices,\
               attr_tokens, attr_in_product, attr_unigram_indices, attr_bigram_indices, attr_char_trigram_indices,\
               features, labels = self.get_batch(batch_size)
            

            return {
                'query_indices': [
                    tf.reshape(query_unigram_indices, [-1, self.maximums_query[0]]), 
                    tf.reshape(query_bigram_indices, [-1, self.maximums_query[1]]), 
                    tf.reshape(query_char_trigram_indices, [-1, self.maximums_query[2]])
                ],
                'product_name_indices': [
                    tf.reshape(product_unigram_indices, [-1, self.maximums_product_name[0]]), 
                    tf.reshape(product_bigram_indices, [-1, self.maximums_product_name[1]]),
                    tf.reshape(product_char_trigram_indices, [-1, self.maximums_product_name[2]])
                ],
                'brand_indices': [
                    tf.reshape(brand_unigram_indices, [-1, self.maximums_brand[0]]),
                    tf.reshape(brand_bigram_indices, [-1, self.maximums_brand[1]]),
                    tf.reshape(brand_char_trigram_indices, [-1, self.maximums_brand[2]])
                ],
                'author_indices': [
                    tf.reshape(author_unigram_indices, [-1, self.maximums_author[0]]),
                    tf.reshape(author_bigram_indices, [-1, self.maximums_author[1]]),
                    tf.reshape(author_char_trigram_indices, [-1, self.maximums_author[2]]),
                ],
                'cat_indices': [
                    tf.reshape(cat_unigram_indices, [-1, self.maximums_cat[0]]),
                    tf.reshape(cat_bigram_indices, [-1, self.maximums_cat[1]]),
                    tf.reshape(cat_char_trigram_indices, [-1, self.maximums_cat[2]])
                ],
                'attr_indices': [
                    tf.reshape(attr_unigram_indices, [-1, self.maximums_attr[0]]),
                    tf.reshape(attr_bigram_indices, [-1, self.maximums_attr[1]]),
                    tf.reshape(attr_char_trigram_indices, [-1, self.maximums_attr[2]])
                ],
                'cat_tokens': cat_tokens,
                'attr_tokens': attr_tokens,
                'cats_in_product': cat_in_product,
                'attrs_in_product': attr_in_product,
                'free_features': tf.reshape(features, [-1, len(self.precomputed)])
            }, labels

        return _inside

    def end(self):
        self.conn.close()