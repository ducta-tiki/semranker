import json
import tensorflow as tf
import pyhash
from tensorflow.python.saved_model import loader
from tensorflow.python.saved_model import tag_constants
from reader.sqlite_product import create_connection, get_product, get_fields
from reader.convert import convert_strings, convert_cats, convert_attrs, convert_features
from vn_lang import query_preprocessing

# https://medium.com/@jsflo.dev/saving-and-loading-a-tensorflow-model-using-the-savedmodel-api-17645576527

class SemRankerPredict:
    def __init__(self, 
        checkpoint_path="export",
        precomputed_path="meta/precomputed.json",
        product_db="db/tiki-products.db",
        vocab_path="meta/vocab.txt",
        cat_tokens_path="meta/cats.txt",
        attr_tokens_path="meta/attrs.txt",
        max_query_length=25,#for unigram, bigram, character trigrams
        max_product_name_length=50, #for unigram, bigram, character trigrams
        max_brand_length=10,
        max_author_length=10,
        max_cat_length=10, #for unigram, bigram, character trigrams
        max_attr_length=10, #for unigram, bigram, character trigram
        unknown_bin=8012
    ):
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

        self.max_query_length = max_query_length
        self.max_product_name_length = max_product_name_length
        self.max_brand_length = max_brand_length
        self.max_author_length = max_author_length
        self.max_cat_length = max_cat_length
        self.max_attr_length = max_attr_length
        
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

        self.conn = create_connection(product_db)
        self.headers = get_fields(self.conn)

        self.sess = tf.Session()
        _ = loader.load(
            self.sess, [tag_constants.SERVING], export_dir=checkpoint_path)

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

    def fit(self, query, pids):
        products = []
        for pid in pids:
            product = self.get_product(str(pid))
            if product:
                products.append(product)

        queries = [query_preprocessing(query),] * len(products)
        product_names = list(map(lambda x: query_preprocessing(x.get("name")), products))
        brands = list(map(lambda x: query_preprocessing(x.get("brand")), products))
        authors = list(map(lambda x: " ".join([query_preprocessing(z) for z in x.get("author")]), products))
        categories = list(map(lambda x: x.get('categories'), products))
        attributes = list(map(lambda x: x.get('attributes'), products))
        features = list(map(lambda x: [float(x.get(h, 0)) for h in self.precomputed], products))
        precomputed_features_min = [self.precomputed.get(h)[0] for h in self.precomputed]
        precomputed_features_max = [self.precomputed.get(h)[1] for h in self.precomputed]
        
        max_query_length = self.max_query_length
        query_unigram_indices, query_bigram_indices, query_char_trigram_indices =  \
            convert_strings(
                queries, self.token_2_idx, self.zero_idx, 
                max_query_length, max_query_length, max_query_length*5, 
                self.unknown_to_idx)
        
        max_product_name_length = self.max_product_name_length
        product_unigram_indices, product_bigram_indices, product_char_trigram_indices =  \
            convert_strings(
                product_names, self.token_2_idx, self.zero_idx, 
                max_product_name_length, max_product_name_length, max_product_name_length*5, 
                self.unknown_to_idx)

        max_brand_length = self.max_brand_length
        brand_unigram_indices, brand_bigram_indices, brand_char_trigram_indices =  \
            convert_strings(
                brands, self.token_2_idx, self.zero_idx, 
                max_brand_length, max_brand_length, max_brand_length*5, 
                self.unknown_to_idx)

        max_author_length = self.max_author_length
        author_unigram_indices, author_bigram_indices, author_char_trigram_indices =  \
            convert_strings(
                authors, self.token_2_idx, self.zero_idx, 
                max_author_length, max_author_length, max_author_length*5, 
                self.unknown_to_idx)

        max_cat_length = self.max_cat_length
        cat_tokens, cat_in_product, cat_unigram_indices, cat_bigram_indices, cat_char_trigram_indices = \
            convert_cats(
                categories,
                self.token_2_idx,
                self.cat_token_2_idx,
                self.zero_idx,
                self.cat_zero_idx,
                self.unknown_to_idx,
                max_cat_length, max_cat_length, max_cat_length*5
            )

        max_attr_length = self.max_attr_length
        attr_tokens, attr_in_product, attr_unigram_indices, attr_bigram_indices, attr_char_trigram_indices = \
            convert_attrs(
                attributes,
                self.token_2_idx,
                self.attr_token_2_idx,
                self.zero_idx,
                self.attr_zero_idx,
                self.unknown_to_idx,
                max_attr_length, max_attr_length, max_attr_length*5
            )

        free_features = convert_features(
            features, precomputed_features_min, precomputed_features_max)

        pred_score = self.sess.run('score:0', feed_dict={
            'query_unigram_indices:0': query_unigram_indices,
            'query_bigram_indices:0':query_bigram_indices, 
            'query_char_trigram_indices:0':query_char_trigram_indices,
            'product_unigram_indices:0': product_unigram_indices,
            'product_bigram_indices:0': product_bigram_indices,
            'product_char_trigram_indices:0': product_char_trigram_indices,
            'brand_unigram_indices:0': brand_unigram_indices,
            'brand_bigram_indices:0': brand_bigram_indices,
            'brand_char_trigram_indices:0': brand_char_trigram_indices,
            'author_unigram_indices:0': author_unigram_indices,
            'author_bigram_indices:0': author_bigram_indices,
            'author_char_trigram_indices:0': author_char_trigram_indices,
            'cat_tokens:0': cat_tokens,
            'cats_in_product:0': cat_in_product,
            'cat_unigram_indices:0': cat_unigram_indices,
            'cat_bigram_indices:0': cat_bigram_indices,
            'cat_char_trigram_indices:0': cat_char_trigram_indices,
            'attr_tokens:0': attr_tokens,
            'attrs_in_product:0': attr_in_product,
            'attr_unigram_indices:0': attr_unigram_indices,
            'attr_bigram_indices:0': attr_bigram_indices,
            'attr_char_trigram_indices:0': attr_char_trigram_indices,
            'free_features:0': free_features
        })

        return list(pred_score), products

if __name__ == "__main__":
    predictor = SemRankerPredict()

    import requests
    from pprint import pprint
    # query = "từ điển điện tử"
    # query = "tv"
    # query = "iphone màu đen"
    # query = "apple watch"
    # query = "realme 3 pro"
    # query = "chuột dây"
    # query = 'xiaomi'
    # query = 'innis free'
    # query = 'máy rung'
    # query = 'dày nữ'
    #query = 'sac du phong iphone'
    #query = 'ma đạo tổ sư đam mỹ'
    # query = 'ủng đi mưa'
    # query = 'đàm thaoij tiếng trung ngành nhà hàng'
    # query = 'vinamil'
    # query = 'lược sử hacker'
    query = 'gel xoa tham quang mat'
    resp = requests.get("http://browser.tiki.services/v2/products?q=%s&limit=500" % query)
    products = list(map(lambda x: x.get("id"), json.loads(resp.text)['data']['data']))
    pred_score, ret_products = predictor.fit(query, products)

    for i, p in enumerate(ret_products):
        p['pos'] = i

    with open('origins.txt', 'w') as fobj:
        json.dump(ret_products, fobj, ensure_ascii=False, indent=2)

    for p, s in zip(ret_products, pred_score):
        p['score'] = float(s)
    
    ret_products = sorted(ret_products, key=lambda x: x['score'], reverse=True)
    for i, p in enumerate(ret_products):
        p['pos'] = i
    with open('ranking.txt', 'w') as fobj:
        json.dump(ret_products, fobj, ensure_ascii=False, indent=2)