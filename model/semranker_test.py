import tensorflow as tf
import numpy as np
from vn_lang import query_preprocessing
from reader.convert import convert_strings, create_ngrams, convert_cats, convert_attrs, convert_features
from model.semranker import SemRanker
from model.loss import semranker_loss
import pyhash


class SemRankerTest(tf.test.TestCase):
    def setUp(self):
        self.unigrams = set()
        self.bigrams = set()
        self.char_trigrams = set()

        self.cat_tokens = set()
        self.attr_tokens = set()

        self.hasher = pyhash.murmur3_32()
        self.unknown_bin = 16

        self.feature_precomputed = {
            "reviews": [0.0, 3437.0], 
            "rating": [0.0, 100.0], 
            "sales_monthly": [0.0, 14345.0], 
            "sales_yearly": [0.0, 136592.0], 
            "support_p2h_delivery": [0.0, 1.0]
        }
        self.header_fields = ["reviews", "rating", "sales_monthly", "sales_yearly", "support_p2h_delivery"]

        p1 = {
            'product_name': 'Ổ Cứng SSD Kingston HyperX FURY 120GB - SATA III - Hàng Chính Hãng',
            'brand': 'Kingston',
            'author': '',
            'attributes': '1165#filter_ssd_storage#120 GB - 128 GB|1166#filter_ssd_product_size#2.5 inch',
            'categories': '1846#2#Laptop - Máy Vi Tính - Linh kiện|8060#3#Thiết bị lưu trữ',
            'reviews': 100,
            'rating': 80,
            'sales_monthly': 20,
            'sales_yearly':100,
            'support_p2h_delivery': 1
        }

        p2 = {
            'product_name': 'Ổ Cứng SSD Sata III 2.5 Inch Samsung 850 EVO 120GB - Hàng Chính Hãng',
            'brand': 'Samsung',
            'author': '',
            'attributes': '1165#filter_ssd_storage#120 GB - 128 GB|1166#filter_ssd_product_size#2.5 inch',
            'categories': '1846#2#Laptop - Máy Vi Tính - Linh kiện|8060#3#Thiết bị lưu trữ',
            'reviews': 71,
            'rating':   95,
            'sales_monthly': 10,
            'sales_yearly':50,
            'support_p2h_delivery': 1
        }

        p3 = {
            'product_name': 'Mai Em Vào Lớp 1 - Vở Tập Tô Chữ (Dành Cho Trẻ 5 - 6 Tuổi) - Tập 1',
            'brand': '',
            'author': ['Lê Hồng Đăng - Lê Thị Ngọc Ánh'],
            'attributes': '',
            'categories': '8322#2#Nhà Sách Tiki|316#3#Sách tiếng Việt |393#4#Sách thiếu nhi |853#5#Kiến thức - Bách khoa',
            'reviews': 0,
            'rating': 0,
            'sales_monthly': 3,
            'sales_yearly': 10,
            'support_p2h_delivery': 1
        }

        self.products = [p1, p2, p3]
        self.queries = ['ổ cứng', 'samsung 850', 'vở tập tô']
        self.target = [2, 1, 0] # positive, impressed, negative

        for p in self.products:
            self.add_to_vocab(query_preprocessing(p['product_name']))

            for z in p['attributes'].split('|'):
                t = "#".join(z.split("#")[:2])
                self.attr_tokens.add(t)

            for z in p['categories'].split('|'):
                t = "#".join(z.split("#")[:2])
                self.cat_tokens.add(t)
        
        self.vocab = self.unigrams.copy()
        self.vocab = self.vocab.union(self.bigrams, self.char_trigrams)
        self.vocab = list(self.vocab)
        self.zero_idx = len(self.vocab) + self.unknown_bin

        self.token_2_idx = {}
        for i, t in enumerate(self.vocab):
            self.token_2_idx[t] = i

        self.cat_token_2_idx = {}
        for i, t in enumerate(self.cat_tokens):
            self.cat_token_2_idx[t] = i
        self.cat_zero_idx = len(self.cat_tokens)

        self.attr_token_2_idx = {}
        for i, t in enumerate(self.attr_tokens):
            self.attr_token_2_idx[t] = i
        self.attr_zero_idx = len(self.attr_tokens)

        self.embed_size = 80
        self.attr_cat_embed_size = 10
        self.vocab_size = len(self.token_2_idx)
        self.max_query_length = 25
        self.max_product_length = 50
        self.max_brand_length = 25
        self.max_author_length = 25
        self.max_attr_length = 10
        self.max_cat_length = 10
        self.filter_sizes = [2,3,4,5]
        self.num_filters = 5

    def add_to_vocab(self, s):
        tokens = s.split()
        for t in tokens:
            self.unigrams.add(t)
            
            z = "#" + t +"#"
            for i in range(0, max(len(z)-3, 0)):
                v = z[i:i+3]
                self.char_trigrams.add(v)

        for i in range(0, max(len(tokens) - 1, 0)):
            t = "%s#%s" % (tokens[i], tokens[i+1])
            self.bigrams.add(t)
            
    def unknown_to_idx(self, unknown):
        return self.hasher(unknown) % self.unknown_bin

    def testCreateNgram(self):
        product_names = list(map(lambda x: query_preprocessing(x.get("product_name")), self.products))
        unigrams, bigrams, char_trigrams = create_ngrams(product_names[0])

        self.assertEqual(unigrams, 
            ['ổ', 'cứng', 'ssd', 'kingston', 'hyperx', 'fury', '120gb', '-', 'sata', 'iii', '-', 'hàng', 'chính', 'hãng']
        )
        self.assertEqual(bigrams,
            ['ổ#cứng', 'cứng#ssd', 'ssd#kingston', 'kingston#hyperx', 'hyperx#fury', 'fury#120gb', '120gb#-', '-#sata', 'sata#iii', 'iii#-', '-#hàng', 'hàng#chính', 'chính#hãng']
        )
        self.assertEqual(char_trigrams,
            ['#ổ#', '#cứ', 'cứn', 'ứng', 'ng#', '#ss', 'ssd', 'sd#', '#ki', 'kin', 'ing', 'ngs', 'gst', 'sto', 'ton', 'on#', '#hy', 'hyp', 'ype', 'per', 'erx', 'rx#', '#fu', 'fur', 'ury', 'ry#', '#12', '120', '20g', '0gb', 'gb#', '#-#', '#sa', 'sat', 'ata', 'ta#', '#ii', 'iii', 'ii#', '#-#', '#hà', 'hàn', 'àng', 'ng#', '#ch', 'chí', 'hín', 'ính', 'nh#', '#hã', 'hãn', 'ãng', 'ng#']
        )

    def testConvert(self):
        query = "ổ cứng hello world"
        unigram_indices, bigram_indices, char_trigram_indices = \
            convert_strings(
                ["ổ cứng hello world"], self.token_2_idx, self.zero_idx, 
                10, 10, 30, self.unknown_to_idx)

        unigrams, bigrams, char_trigrams = create_ngrams(query)

        self.assertEqual(len(unigram_indices[0]), 10)
        self.assertEqual(len(bigram_indices[0]), 10)
        self.assertEqual(len(char_trigram_indices[0]), 30)

        base_unknown = len(self.token_2_idx)

        for ngram, indices in zip(
            [unigrams, bigrams, char_trigrams], [unigram_indices, bigram_indices, char_trigram_indices]):
            for i, t in enumerate(ngram):
                if t in self.token_2_idx:
                    self.assertEqual(self.token_2_idx[t], indices[0][i])
                else:
                    self.assertEqual(base_unknown + self.unknown_to_idx(t), indices[0][i])

            for t in indices[0][len(ngram):]:
                self.assertEqual(t, self.zero_idx)

        cat_indices, cat_in_product, unigram_indices, bigram_indices, char_trigram_indices = \
            convert_cats(
                [
                    '8322#2#Nhà Sách Tiki|316#3#Sách tiếng Việt |393#4#Sách thiếu nhi |853#5#Kiến thức - Bách khoa',
                    '1846#2#Laptop - Máy Vi Tính - Linh kiện|8060#3#Thiết bị lưu trữ'
                ],
                self.token_2_idx,
                self.cat_token_2_idx,
                self.zero_idx,
                self.cat_zero_idx,
                self.unknown_to_idx,
                10, 10, 50
            )
        actual_cats = ['8322#2', '316#3', '393#4', '853#5', '1846#2', '8060#3']
        for cat_idx, ac in zip(cat_indices, actual_cats):
            self.assertEqual(cat_idx, self.cat_token_2_idx[ac])
        
        self.assertEqual(list(cat_in_product), [4, 2])

        u1, b1, c1 = convert_strings(
            ["nhà sách tiki"], self.token_2_idx, self.zero_idx, 
            10, 10, 50, self.unknown_to_idx)
        u2, b2, c2 = convert_strings(
            ["sách tiếng việt"], self.token_2_idx, self.zero_idx, 
            10, 10, 50, self.unknown_to_idx)
        un, bn, cn = convert_strings(
            ["thiết bị lưu trữ"], self.token_2_idx, self.zero_idx, 
            10, 10, 50, self.unknown_to_idx)

        self.assertEqual(list(unigram_indices[0]), list(u1[0]))
        self.assertEqual(list(bigram_indices[0]), list(b1[0]))
        self.assertEqual(list(char_trigram_indices[0]), list(c1[0]))

        self.assertEqual(list(unigram_indices[1]), list(u2[0]))
        self.assertEqual(list(bigram_indices[1]), list(b2[0]))
        self.assertEqual(list(char_trigram_indices[1]), list(c2[0]))

        self.assertEqual(list(unigram_indices[-1]), list(un[0]))
        self.assertEqual(list(bigram_indices[-1]), list(bn[0]))
        self.assertEqual(list(char_trigram_indices[-1]), list(cn[0]))

        attr_indices, attr_in_product, unigram_indices, bigram_indices, char_trigram_indices = \
            convert_attrs(
                [
                    '1165#filter_ssd_storage#120 GB - 128 GB|1166#filter_ssd_product_size#2.5 inch',
                    ''
                ],
                self.token_2_idx,
                self.attr_token_2_idx,
                self.zero_idx,
                self.attr_zero_idx,
                self.unknown_to_idx,
                10, 10, 50
            )

        actual_attrs = ['1165#filter_ssd_storage', '1166#filter_ssd_product_size']
        for attr_idx, ac in zip(attr_indices[:2], actual_attrs):
            self.assertEqual(attr_idx, self.attr_token_2_idx[ac])
        self.assertEqual(attr_indices[2], self.attr_zero_idx)
        
        u1, b1, c1 = convert_strings(
            ["120 gb - 128 gb"], self.token_2_idx, self.zero_idx, 
            10, 10, 50, self.unknown_to_idx)
        u2, b2, c2 = convert_strings(
            ["2.5 inch"], self.token_2_idx, self.zero_idx, 
            10, 10, 50, self.unknown_to_idx)
        
        self.assertEqual(list(unigram_indices[0]), list(u1[0]))
        self.assertEqual(list(bigram_indices[0]), list(b1[0]))
        self.assertEqual(list(char_trigram_indices[0]), list(c1[0]))

        self.assertEqual(list(unigram_indices[1]), list(u2[0]))
        self.assertEqual(list(bigram_indices[1]), list(b2[0]))
        self.assertEqual(list(char_trigram_indices[1]), list(c2[0]))
        
        self.assertEqual(list(unigram_indices[-1]), [self.zero_idx,]*10)
        self.assertEqual(list(bigram_indices[-1]), [self.zero_idx,]*10)
        self.assertEqual(list(char_trigram_indices[-1]), [self.zero_idx,]*50)

    def create_placeholder_data(self):
        queries = list(map(lambda x: query_preprocessing(x), self.queries))
        product_names = list(map(lambda x: query_preprocessing(x.get("product_name")), self.products))
        brands = list(map(lambda x: query_preprocessing(x.get("brand")), self.products))
        authors = list(map(lambda x: " ".join([query_preprocessing(z) for z in x.get("author")]), self.products))
        categories = list(map(lambda x: x.get('categories'), self.products))
        attributes = list(map(lambda x: x.get('attributes'), self.products))
        features = list(map(lambda x: [x.get(h) for h in self.header_fields], self.products))
        precomputed_features_min = [self.feature_precomputed.get(h)[0] for h in self.header_fields]
        precomputed_features_max = [self.feature_precomputed.get(h)[1] for h in self.header_fields]
        
        max_query_length = self.max_query_length
        query_unigram_indices, query_bigram_indices, query_char_trigram_indices =  \
            convert_strings(
                queries, self.token_2_idx, self.zero_idx, 
                max_query_length, max_query_length, max_query_length*5, 
                self.unknown_to_idx)
        
        max_product_length = self.max_product_length
        product_unigram_indices, product_bigram_indices, product_char_trigram_indices =  \
            convert_strings(
                product_names, self.token_2_idx, self.zero_idx, 
                max_product_length, max_product_length, max_product_length*5, 
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

        features = convert_features(
            features, precomputed_features_min, precomputed_features_max)

        return query_unigram_indices, query_bigram_indices, query_char_trigram_indices, \
               product_unigram_indices, product_bigram_indices, product_char_trigram_indices, \
               brand_unigram_indices, brand_bigram_indices, brand_char_trigram_indices, \
               author_unigram_indices, author_bigram_indices, author_char_trigram_indices, \
               cat_tokens, cat_in_product, cat_unigram_indices, cat_bigram_indices, cat_char_trigram_indices,\
               attr_tokens, attr_in_product, attr_unigram_indices, attr_bigram_indices, attr_char_trigram_indices,\
               features

    def init_graph(self):
        embed_size = self.embed_size
        vocab_size = self.vocab_size
        max_query_length = self.max_query_length
        max_product_length = self.max_product_length
        max_brand_length = self.max_brand_length
        max_author_length = self.max_author_length
        max_cat_length = self.max_cat_length
        max_attr_length = self.max_attr_length

        ranker = SemRanker(
            vocab_size=vocab_size,
            unknown_bin=self.unknown_bin,
            cat_tokens_size=len(self.cat_tokens),
            attr_tokens_size=len(self.attr_tokens),
            embed_size=embed_size,
            attr_cat_embed_size=self.attr_cat_embed_size,
            filter_sizes=self.filter_sizes,
            max_query_length=max_query_length,
            max_product_name_length=max_product_length,
            max_brand_length=max_brand_length,
            max_author_length=max_author_length,
            max_attr_length=self.max_attr_length,
            max_cat_length=self.max_cat_length, 
            num_filters=self.num_filters
        )

        query_unigram_indices = tf.placeholder(
            tf.int32, shape=[None, max_query_length], name="query_unigram_indices")
        query_bigram_indices = tf.placeholder(
            tf.int32, shape=[None, max_query_length], name="query_bigram_indices")
        query_char_trigram_indices = tf.placeholder(
            tf.int32, shape=[None, max_query_length*5], name="query_char_trigram_indices")
        product_unigram_indices = tf.placeholder(
            tf.int32, shape=[None, max_product_length], name="product_unigram_indices")
        product_bigram_indices = tf.placeholder(
            tf.int32, shape=[None, max_product_length], name="product_bigram_indices")
        product_char_trigram_indices = tf.placeholder(
            tf.int32, shape=[None, max_product_length*5], name="product_char_trigram_indices")
        brand_unigram_indices = tf.placeholder(
            tf.int32, shape=[None, max_brand_length], name="brand_unigram_indices")
        brand_bigram_indices = tf.placeholder(
            tf.int32, shape=[None, max_brand_length], name="brand_bigram_indices")
        brand_char_trigram_indices = tf.placeholder(
            tf.int32, shape=[None, max_brand_length*5], name="brand_char_trigram_indices")
        author_unigram_indices = tf.placeholder(
            tf.int32, shape=[None, max_author_length], name="author_unigram_indices")
        author_bigram_indices = tf.placeholder(
            tf.int32, shape=[None, max_author_length], name="author_bigram_indices")
        author_char_trigram_indices = tf.placeholder(
            tf.int32, shape=[None, max_author_length*5], name="author_char_trigram_indices")

        cat_unigram_indices = tf.placeholder(
            tf.int32, shape=[None, max_cat_length], name="cat_unigram_indices")
        cat_bigram_indices = tf.placeholder(
            tf.int32, shape=[None, max_cat_length], name="cat_bigram_indices")
        cat_char_trigram_indices = tf.placeholder(
            tf.int32, shape=[None, max_cat_length*5], name="cat_char_trigram_indices")
        cat_tokens = tf.placeholder(tf.int32, shape=[None,], name="cat_tokens")
        cats_in_product = tf.placeholder(tf.int32, shape=[None,], name="cats_in_product")

        attr_unigram_indices = tf.placeholder(
            tf.int32, shape=[None, max_attr_length], name="attr_unigram_indices")
        attr_bigram_indices = tf.placeholder(
            tf.int32, shape=[None, max_attr_length], name="attr_bigram_indices")
        attr_char_trigram_indices = tf.placeholder(
            tf.int32, shape=[None, max_attr_length*5], name="attr_char_trigram_indices")
        attr_tokens = tf.placeholder(tf.int32, shape=[None,], name="attr_tokens")
        attrs_in_product = tf.placeholder(tf.int32, shape=[None,], name="attrs_in_product")

        free_features = tf.placeholder(tf.float32, shape=[None, len(self.header_fields)], name="free_features")

        inputs = {
            'query_unigram_indices': query_unigram_indices,
            'query_bigram_indices': query_bigram_indices,
            'query_char_trigram_indices': query_char_trigram_indices,
            'product_unigram_indices': product_unigram_indices,
            'product_bigram_indices': product_bigram_indices,
            'product_char_trigram_indices': product_char_trigram_indices,
            'brand_unigram_indices': brand_unigram_indices,
            'brand_bigram_indices':brand_bigram_indices,
            'brand_char_trigram_indices': brand_char_trigram_indices,
            'author_unigram_indices': author_unigram_indices,
            'author_bigram_indices': author_bigram_indices,
            'author_char_trigram_indices': author_char_trigram_indices,
            'cat_unigram_indices': cat_unigram_indices,
            'cat_bigram_indices': cat_bigram_indices,
            'cat_char_trigram_indices': cat_char_trigram_indices,
            'attr_unigram_indices': attr_unigram_indices,
            'attr_bigram_indices': attr_bigram_indices,
            'attr_char_trigram_indices': attr_char_trigram_indices,
            'cat_tokens': cat_tokens,
            'attr_tokens': attr_tokens,
            'cat_in_product': cats_in_product,
            'attr_in_product': attrs_in_product,
            'features': free_features,
        }

        score = ranker(
            **inputs
        )

        inputs['number_of_queries'] = 3
        inputs['score'] = score
        return inputs

    def testEmbeddingShapes(self):
        tf.reset_default_graph()
        self.init_graph()
        embed_size = self.embed_size
        attr_cat_embed_size = self.attr_cat_embed_size
        vocab_size = len(self.token_2_idx)

        init_op = tf.initializers.global_variables()
        with self.cached_session() as sess:
            sess.run(init_op)
            n_gram_embedding = sess.run('embedding/zero_padding_n_gram_embedding:0')
            self.assertEqual(n_gram_embedding.shape, (vocab_size+self.unknown_bin+1, embed_size))
            np.testing.assert_array_equal(
                n_gram_embedding[vocab_size+self.unknown_bin:].flatten(), np.zeros([embed_size,]))

            cat_embedding = sess.run('embedding/zero_padding_cat_embedding:0')
            self.assertEqual(cat_embedding.shape, (len(self.cat_tokens)+1, attr_cat_embed_size))
            np.testing.assert_array_equal(
                cat_embedding[len(self.cat_tokens):].flatten(), np.zeros([attr_cat_embed_size,]))

            attr_embedding = sess.run('embedding/zero_padding_attr_embedding:0')
            self.assertEqual(attr_embedding.shape, (len(self.attr_tokens)+1, attr_cat_embed_size))
            np.testing.assert_array_equal(
                attr_embedding[len(self.attr_tokens):].flatten(), np.zeros([attr_cat_embed_size,]))

    def testLayerShapes(self):
        tf.reset_default_graph()
        inputs = self.init_graph()

        query_unigram_indices, query_bigram_indices, query_char_trigram_indices, \
        product_unigram_indices, product_bigram_indices, product_char_trigram_indices, \
        brand_unigram_indices, brand_bigram_indices, brand_char_trigram_indices, \
        author_unigram_indices, author_bigram_indices, author_char_trigram_indices, \
        cat_tokens, cats_in_product, cat_unigram_indices, cat_bigram_indices, cat_char_trigram_indices, \
        attr_tokens, attrs_in_product, attr_unigram_indices, attr_bigram_indices, attr_char_trigram_indices, \
        free_features = \
            self.create_placeholder_data()

        init_op = tf.initializers.global_variables()
        with self.cached_session() as sess:
            sess.run(init_op)

            query_encode, product_name_encode, brand_encode, \
            author_encode, category_encode, attribute_encode, \
            product_encode, score = \
                sess.run([
                    'calc_embedding_feature/query_encode:0','calc_embedding_feature/product_name_encode:0',
                    'calc_embedding_feature/brand_encode:0','calc_embedding_feature/author_encode:0',
                    'calc_embedding_feature/category_encode:0','calc_embedding_feature/attribute_encode:0',
                    'bn/product_encode:0', 'score:0'], 
                    feed_dict={
                        inputs['query_unigram_indices']: query_unigram_indices,
                        inputs['query_bigram_indices']: query_bigram_indices,
                        inputs['query_char_trigram_indices']: query_char_trigram_indices,
                        inputs['product_unigram_indices']: product_unigram_indices,
                        inputs['product_bigram_indices']: product_bigram_indices,
                        inputs['product_char_trigram_indices']: product_char_trigram_indices,
                        inputs['brand_unigram_indices']: brand_unigram_indices,
                        inputs['brand_bigram_indices']: brand_bigram_indices,
                        inputs['brand_char_trigram_indices']: brand_char_trigram_indices,
                        inputs['author_unigram_indices']: author_unigram_indices,
                        inputs['author_bigram_indices']: author_bigram_indices,
                        inputs['author_char_trigram_indices']: author_char_trigram_indices,
                        inputs['cat_tokens']: cat_tokens,
                        inputs['cat_in_product']: cats_in_product,
                        inputs['cat_unigram_indices']: cat_unigram_indices,
                        inputs['cat_bigram_indices']: cat_bigram_indices,
                        inputs['cat_char_trigram_indices']: cat_char_trigram_indices,
                        inputs['attr_tokens']: attr_tokens,
                        inputs['attr_in_product']: attrs_in_product,
                        inputs['attr_unigram_indices']: attr_unigram_indices,
                        inputs['attr_bigram_indices']: attr_bigram_indices,
                        inputs['attr_char_trigram_indices']: attr_char_trigram_indices,
                        inputs['features']: free_features
                    })
            self.assertEqual(
                query_encode.shape, 
                (len(self.queries), len(self.filter_sizes)*self.num_filters*3)
            )

            self.assertEqual(
                product_name_encode.shape, 
                (len(self.products), len(self.filter_sizes)*self.num_filters*3)
            )

            self.assertEqual(
                brand_encode.shape, 
                (len(self.products), len(self.filter_sizes)*self.num_filters*3)
            )

            self.assertEqual(
                author_encode.shape, 
                (len(self.products), len(self.filter_sizes)*self.num_filters*3)
            )

            self.assertEqual(
                category_encode.shape, 
                (len(self.products), (len(self.filter_sizes)*self.num_filters + self.attr_cat_embed_size)*3)
            )

            self.assertEqual(
                attribute_encode.shape, 
                (len(self.products), (len(self.filter_sizes)*self.num_filters + self.attr_cat_embed_size)*3)
            )

            self.assertEqual(
                product_encode.shape, 
                (len(self.products), product_name_encode.shape[1] + 
                                brand_encode.shape[1] + author_encode.shape[1] + 
                                category_encode.shape[1] + attribute_encode.shape[1] + 
                                free_features.shape[1])
            )

            self.assertEqual(score.shape, (len(self.products),))

    def testOverfit(self):
        tf.reset_default_graph()
        inputs = self.init_graph()

        loss = semranker_loss(self.target, inputs['score'] , tf.cast(inputs['number_of_queries'], tf.float32))

        global_step = tf.train.get_or_create_global_step()
        opt = tf.train.MomentumOptimizer(
            learning_rate=0.1,
            momentum=0.9
        )
        grads = opt.compute_gradients(loss)
        train_op = opt.apply_gradients(grads, global_step=global_step)
        update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
        train_op = tf.group([train_op, update_ops])

        query_unigram_indices, query_bigram_indices, query_char_trigram_indices, \
        product_unigram_indices, product_bigram_indices, product_char_trigram_indices, \
        brand_unigram_indices, brand_bigram_indices, brand_char_trigram_indices, \
        author_unigram_indices, author_bigram_indices, author_char_trigram_indices, \
        cat_tokens, cats_in_product, cat_unigram_indices, cat_bigram_indices, cat_char_trigram_indices, \
        attr_tokens, attrs_in_product, attr_unigram_indices, attr_bigram_indices, attr_char_trigram_indices, \
        free_features = \
            self.create_placeholder_data()

        init_op = tf.initializers.global_variables()
        with self.cached_session() as sess:
            sess.run(init_op)

            max_iter = 100
            ret_loss, ret_score =  [], []
            for _ in range(max_iter):
                _, ret_loss, ret_score = sess.run([train_op, loss, inputs['score']], 
                    feed_dict={
                        inputs['query_unigram_indices']: query_unigram_indices,
                        inputs['query_bigram_indices']: query_bigram_indices,
                        inputs['query_char_trigram_indices']: query_char_trigram_indices,
                        inputs['product_unigram_indices']: product_unigram_indices,
                        inputs['product_bigram_indices']: product_bigram_indices,
                        inputs['product_char_trigram_indices']: product_char_trigram_indices,
                        inputs['brand_unigram_indices']: brand_unigram_indices,
                        inputs['brand_bigram_indices']: brand_bigram_indices,
                        inputs['brand_char_trigram_indices']: brand_char_trigram_indices,
                        inputs['author_unigram_indices']: author_unigram_indices,
                        inputs['author_bigram_indices']: author_bigram_indices,
                        inputs['author_char_trigram_indices']: author_char_trigram_indices,
                        inputs['cat_tokens']: cat_tokens,
                        inputs['cat_in_product']: cats_in_product,
                        inputs['cat_unigram_indices']: cat_unigram_indices,
                        inputs['cat_bigram_indices']: cat_bigram_indices,
                        inputs['cat_char_trigram_indices']: cat_char_trigram_indices,
                        inputs['attr_tokens']: attr_tokens,
                        inputs['attr_in_product']: attrs_in_product,
                        inputs['attr_unigram_indices']: attr_unigram_indices,
                        inputs['attr_bigram_indices']: attr_bigram_indices,
                        inputs['attr_char_trigram_indices']: attr_char_trigram_indices,
                        inputs['features']: free_features
                    })

                print("Loss:%0.4f" % float(ret_loss), list(ret_score))
            self.assertGreaterEqual(ret_score[0], 0.9)

if __name__ == "__main__":
    tf.test.main()