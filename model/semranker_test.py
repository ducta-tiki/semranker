import tensorflow as tf
import numpy as np
from vn_lang import query_preprocessing
from reader.convert import convert_strings, create_ngrams, convert_cats, convert_attrs
from model.semranker import SemRanker
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

        feature_precomputed = {
            "reviews": [0.0, 3437.0], 
            "rating": [0.0, 100.0], 
            "sales_monthly": [0.0, 0.0], 
            "sales_yearly": [0.0, 0.0], 
            "support_p2h_delivery": [0.0, 1.0]
        }

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
        self.queries = ['ổ cứng', 'vở tập tô', 'ốp lưng']

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
        self.vocab_size = len(self.token_2_idx)
        self.max_query_length = 25
        self.max_product_length = 50
        self.max_brand_length = 25
        self.max_author_length = 25
        self.max_attr_length = 20
        self.max_cat_length = 20
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

        convert_cats()

    def create_placeholder_data(self):
        queries = list(map(lambda x: query_preprocessing(x), self.queries))
        product_names = list(map(lambda x: query_preprocessing(x.get("product_name")), self.products))
        brands = list(map(lambda x: query_preprocessing(x.get("brand")), self.products))
        authors = list(map(lambda x: " ".join([query_preprocessing(z) for z in x.get("author")]), self.products))

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

        return np.asarray(query_unigram_indices, dtype=np.int32), \
               np.asarray(query_bigram_indices, dtype=np.int32), \
               np.asarray(query_char_trigram_indices, dtype=np.int32), \
               np.asarray(product_unigram_indices, dtype=np.int32), \
               np.asarray(product_bigram_indices, dtype=np.int32), \
               np.asarray(product_char_trigram_indices, dtype=np.int32),\
               np.asarray(brand_unigram_indices, dtype=np.int32), \
               np.asarray(brand_bigram_indices, dtype=np.int32), \
               np.asarray(brand_char_trigram_indices, dtype=np.int32), \
               np.asarray(author_unigram_indices, dtype=np.int32), \
               np.asarray(author_bigram_indices, dtype=np.int32), \
               np.asarray(author_char_trigram_indices, dtype=np.int32)

    def init_graph(self):
        embed_size = self.embed_size
        vocab_size = self.vocab_size
        max_query_length = self.max_query_length
        max_product_length = self.max_product_length
        max_brand_length = self.max_brand_length
        max_author_length = self.max_author_length

        ranker = SemRanker(
            vocab_size=vocab_size,
            unknown_bin=self.unknown_bin,
            cat_tokens_size=len(self.cat_tokens),
            attr_tokens_size=len(self.attr_tokens),
            embed_size=embed_size,
            filter_sizes=self.filter_sizes,
            max_query_length=max_query_length,
            max_product_name_length=self.max_product_length,
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

        ranker(
            query_indices=[query_unigram_indices, query_bigram_indices, query_char_trigram_indices],
            product_name_indices=[product_unigram_indices, product_bigram_indices, product_char_trigram_indices],
            brand_indices=[brand_unigram_indices, brand_bigram_indices, brand_char_trigram_indices],
            author_indices=[author_unigram_indices, author_bigram_indices, author_char_trigram_indices]
        )

        return {
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
            'author_char_trigram_indices': author_char_trigram_indices
        }

    def testEmbeddingShapes(self):
        tf.reset_default_graph()
        self.init_graph()
        embed_size = 80
        vocab_size = len(self.token_2_idx)

        init_op = tf.initializers.global_variables()
        with self.cached_session() as sess:
            sess.run(init_op)
            n_gram_embedding = sess.run('embedding/zero_padding_n_gram_embedding:0')
            self.assertEqual(n_gram_embedding.shape, (vocab_size+self.unknown_bin+1, embed_size))
            np.testing.assert_array_equal(
                n_gram_embedding[vocab_size+self.unknown_bin:].flatten(), np.zeros([embed_size,]))

            cat_embedding = sess.run('embedding/zero_padding_cat_embedding:0')
            self.assertEqual(cat_embedding.shape, (len(self.cat_tokens)+1, embed_size))
            np.testing.assert_array_equal(
                cat_embedding[len(self.cat_tokens):].flatten(), np.zeros([embed_size,]))

            attr_embedding = sess.run('embedding/zero_padding_attr_embedding:0')
            self.assertEqual(attr_embedding.shape, (len(self.attr_tokens)+1, embed_size))
            np.testing.assert_array_equal(
                attr_embedding[len(self.attr_tokens):].flatten(), np.zeros([embed_size,]))

    def testTextCNNLayer(self):
        tf.reset_default_graph()
        inputs = self.init_graph()

        query_unigram_indices, query_bigram_indices, query_char_trigram_indices, \
        product_unigram_indices, product_bigram_indices, product_char_trigram_indices, \
        brand_unigram_indices, brand_bigram_indices, brand_char_trigram_indices, \
        author_unigram_indices, author_bigram_indices, author_char_trigram_indices = \
             self.create_placeholder_data()

        init_op = tf.initializers.global_variables()
        with self.cached_session() as sess:
            sess.run(init_op)

            query_encode, product_encode, brand_encode, author_encode = sess.run([
                'calc_embedding_feature/query_encode:0','calc_embedding_feature/product_encode:0',
                'calc_embedding_feature/brand_encode:0','calc_embedding_feature/author_encode:0'], 
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
                    inputs['author_char_trigram_indices']: author_char_trigram_indices
                })
            self.assertEqual(
                query_encode.shape, 
                (len(self.queries), len(self.filter_sizes)*self.num_filters*3)
            )

            self.assertEqual(
                product_encode.shape, 
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

if __name__ == "__main__":
    tf.test.main()