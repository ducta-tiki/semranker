import tensorflow as tf

# https://github.com/dennybritz/cnn-text-classification-tf/blob/master/text_cnn.py  

class SemRanker(object):
    def __init__(
        self, vocab_size, unknown_bin, cat_tokens_size, attr_tokens_size, 
        embed_size=80, attr_cat_embed_size=10, filter_sizes=[2,3,4,5], 
        max_query_length=40, max_product_name_length=50, 
        max_brand_length=25, max_author_length=25,
        max_attr_length=20, max_cat_length=20, num_filters=5):
        self.embed_size = embed_size
        self.attr_cat_embed_size = attr_cat_embed_size
        self.max_query_length = max_query_length
        self.max_product_name_length = max_product_name_length
        self.max_brand_length = max_brand_length
        self.max_author_length = max_author_length
        self.max_attr_length = max_attr_length
        self.max_cat_length = max_cat_length
        self.filter_sizes = filter_sizes
        self.num_filters = num_filters
        self.vocab_size = vocab_size
        self.unknown_bin = unknown_bin
        self.cat_tokens_size = cat_tokens_size
        self.attr_tokens_size = attr_tokens_size
        self.pos_embed_size = 100

    def reduce_in_product(self, h, tokens_embed, in_product):
        h_pool_flat = tf.concat([h, tokens_embed], axis=1) # sum_of(in_product) x (embed_size + token_embed_size)
        rz = tf.reduce_sum(in_product)
        z1 = tf.cumsum(in_product, exclusive=True)
        z2 = tf.cumsum(in_product)
        mask = tf.sequence_mask(z2, maxlen=rz, dtype=tf.float32) - \
            tf.sequence_mask(z1, maxlen=rz, dtype=tf.float32) # batch_size x sum_of(in_product)
        p = tf.matmul(mask, h_pool_flat)
        p = p / tf.expand_dims(tf.cast(in_product, tf.float32), axis=1) # batch_size x (embed_size + token_embed_size)
        return p

    def __call__(
        self, query_unigram_indices=None, query_bigram_indices=None, query_char_trigram_indices=None,
        product_unigram_indices=None, product_bigram_indices=None, product_char_trigram_indices=None,
        brand_unigram_indices=None, brand_bigram_indices=None, brand_char_trigram_indices=None,
        author_unigram_indices=None, author_bigram_indices=None, author_char_trigram_indices=None,
        cat_tokens=None, cat_in_product=None, cat_unigram_indices=None, cat_bigram_indices=None, 
        cat_char_trigram_indices=None, attr_tokens=None, attr_in_product=None, attr_unigram_indices=None,
        attr_bigram_indices=None, attr_char_trigram_indices=None, features=None, qids=None, training=True
    ):
        query_indices = [query_unigram_indices, query_bigram_indices, query_char_trigram_indices]
        product_name_indices = [product_unigram_indices, product_bigram_indices, product_char_trigram_indices]
        brand_indices = [brand_unigram_indices, brand_bigram_indices, brand_char_trigram_indices]
        author_indices = [author_unigram_indices, author_bigram_indices, author_char_trigram_indices]
        cat_indices = [cat_unigram_indices, cat_bigram_indices, cat_char_trigram_indices]
        attr_indices = [ attr_unigram_indices, attr_bigram_indices, attr_char_trigram_indices]

        with tf.variable_scope("embedding"):
            pos_query_unigram_embed = tf.get_variable(
                'pos_query_unigram_embedding', [self.max_query_length, self.pos_embed_size],
                initializer=tf.random_uniform_initializer(minval=-1., maxval=1.)
            )
            pos_query_bigram_embed = tf.get_variable(
                'pos_query_bigram_embedding', [self.max_query_length, self.pos_embed_size],
                initializer=tf.random_uniform_initializer(minval=-1., maxval=1.)
            )
            pos_query_trigram_embed = tf.get_variable(
                'pos_query_trigram_embedding', [self.max_query_length*5, self.pos_embed_size],
                initializer=tf.random_uniform_initializer(minval=-1., maxval=1.)
            )

            pos_product_name_unigram_embed = tf.get_variable(
                'pos_product_name_unigram_embedding', [self.max_product_name_length, self.pos_embed_size],
                initializer=tf.random_uniform_initializer(minval=-1., maxval=1.)
            )
            pos_product_name_bigram_embed = tf.get_variable(
                'pos_product_name_bigram_embedding', [self.max_product_name_length, self.pos_embed_size],
                initializer=tf.random_uniform_initializer(minval=-1., maxval=1.)
            )
            pos_product_name_trigram_embed = tf.get_variable(
                'pos_product_name_trigram_embedding', [self.max_product_name_length*5, self.pos_embed_size],
                initializer=tf.random_uniform_initializer(minval=-1., maxval=1.)
            )

            ngram_embed_weights = tf.get_variable(
                'n_gram_embedding', [self.vocab_size+self.unknown_bin, self.embed_size],
                initializer=tf.random_uniform_initializer(minval=-1., maxval=1.)
            )
            zero_vector_ngram = tf.zeros([1, self.embed_size], dtype=tf.float32)
            ngram_embed_weights = tf.concat(
                [ngram_embed_weights, zero_vector_ngram], axis=0, name="zero_padding_n_gram_embedding")

            zero_vector_attr_cat = tf.zeros([1, self.attr_cat_embed_size], dtype=tf.float32)
            cat_embed_weights = tf.get_variable(
                'cat_embedding', [self.cat_tokens_size, self.attr_cat_embed_size]
            )
            cat_embed_weights = tf.concat(
                [cat_embed_weights, zero_vector_attr_cat], axis=0, name="zero_padding_cat_embedding")

            attr_embed_weights = tf.get_variable(
                'attr_embedding', [self.attr_tokens_size, self.attr_cat_embed_size]
            )
            attr_embed_weights = tf.concat(
                [attr_embed_weights, zero_vector_attr_cat], axis=0, name="zero_padding_attr_embedding")

        with tf.name_scope("calc_embedding_feature"):
            embed_queries = []
            for qz, name, pos, max_sq in zip(query_indices, ['unigram', 'bigram', 'char_trigram'], 
                        [pos_query_unigram_embed, pos_query_bigram_embed, pos_query_trigram_embed],
                        [self.max_query_length, self.max_query_length, self.max_query_length*5]):
                qz_embed = tf.nn.embedding_lookup(ngram_embed_weights, qz) #batch_size x max_query_length x embed_size
                batch_size = tf.shape(qz_embed)[0]
                t_pos = tf.tile(pos, [batch_size, 1])
                t_pos = tf.reshape(t_pos, [batch_size, max_sq, self.pos_embed_size])
                qz_embed = tf.concat([qz_embed, t_pos], axis=2)
                embed_queries.append(
                    tf.reduce_mean(tf.transpose(qz_embed, [0, 2, 1]), axis=2))

            embed_queries = (embed_queries[0] + embed_queries[1] + embed_queries[2]) / 3.
            embed_queries = tf.identity(embed_queries, name="query_encode")

            embed_product_names = []
            for pz, name, pos, max_sq in zip(product_name_indices, ['unigram', 'bigram', 'char_trigram'],
                [pos_product_name_unigram_embed, pos_product_name_bigram_embed, pos_product_name_trigram_embed],
                [self.max_product_name_length, self.max_product_name_length, self.max_product_name_length*5]):
                pz_embed = tf.nn.embedding_lookup(ngram_embed_weights, pz)
                batch_size = tf.shape(pz)[0]
                t_pos = tf.tile(pos, [batch_size, 1])
                t_pos = tf.reshape(t_pos, [batch_size, max_sq, self.pos_embed_size])
                pz_embed = tf.concat([pz_embed, t_pos], axis=2)
                embed_product_names.append(
                    tf.reduce_mean(tf.transpose(pz_embed, [0, 2, 1]), axis=2))

            embed_product_names = (embed_product_names[0] + embed_product_names[1] + embed_product_names[2]) / 3.
            embed_product_names = tf.identity(embed_product_names, name="product_name_encode")

            embed_brand = []
            for bz, name in zip(brand_indices, ['unigram', 'bigram', 'char_trigram']):
                bz_embed = tf.nn.embedding_lookup(ngram_embed_weights, bz)
                embed_brand.append(
                    tf.reduce_mean(tf.transpose(bz_embed, [0, 2, 1]), axis=2))

            embed_brand = (embed_brand[0] + embed_brand[1] + embed_brand[2]) / 3.
            embed_brand = tf.identity(embed_brand, name="brand_encode")

            embed_author = []
            for az, name in zip(author_indices, ['unigram', 'bigram', 'char_trigram']):
                az_embed = tf.nn.embedding_lookup(ngram_embed_weights, az)
                embed_author.append(
                    tf.reduce_mean(tf.transpose(az_embed, [0, 2, 1]), axis=2))

            embed_author = (embed_author[0] + embed_author[1] + embed_author[2]) / 3.
            embed_author = tf.identity(embed_author, name="author_encode")

            embed_cat_tokens = tf.nn.embedding_lookup(cat_embed_weights, cat_tokens)
            embed_cat = []
            for cz, name, scale in zip(cat_indices, ['unigram', 'bigram', 'char_trigram'], [1,1,5]):
                cz_embed = tf.nn.embedding_lookup(ngram_embed_weights, cz)
                embed_cat.append(
                    tf.reduce_mean(tf.transpose(cz_embed, [0, 2, 1]), axis=2))

            embed_cat = (embed_cat[0] + embed_cat[1] + embed_cat[2]) / 3.
            embed_cat = tf.identity(embed_cat, name="category_encode")
            embed_cat = self.reduce_in_product(embed_cat, embed_cat_tokens, cat_in_product)

            embed_attr_tokens = tf.nn.embedding_lookup(attr_embed_weights, attr_tokens)
            embed_attr = []
            for attr_z, name, scale in zip(attr_indices, ['unigram', 'bigram', 'char_trigram'], [1,1,5]):
                attr_z_embed = tf.nn.embedding_lookup(ngram_embed_weights, attr_z)
                embed_attr.append(
                    tf.reduce_mean(tf.transpose(attr_z_embed, [0, 2, 1]), axis=2))

            embed_attr = (embed_attr[0] + embed_attr[1] + embed_attr[2]) / 3.
            embed_attr = tf.identity(embed_attr, name="attribute_encode")
            embed_attr = self.reduce_in_product(embed_attr, embed_attr_tokens, attr_in_product)

            embed_cat = tf.reshape(embed_cat, 
                [-1, self.embed_size + self.attr_cat_embed_size])
            embed_attr= tf.reshape(embed_cat, 
                [-1, self.embed_size + self.attr_cat_embed_size])
            product_features = tf.concat([
                embed_product_names, embed_brand, embed_author, embed_cat, embed_attr, features
            ], axis=1)


        with tf.name_scope("bn"):
            
            v1 = tf.layers.dense(
                product_features,
                units=self.embed_size+self.pos_embed_size,
                kernel_initializer=tf.random_normal_initializer(mean=0, stddev=0.01),
                activation=tf.nn.tanh
            )
            vl = tf.layers.Dense(
                self.embed_size+self.pos_embed_size, activation=tf.nn.tanh, name="common_dense")

            product_features = vl(v1)
            product_features = tf.layers.batch_normalization(
                                v1, training=training)
            product_features = tf.identity(product_features, name="product_encode")

            query_features = vl(embed_queries)
            
            query_features = tf.layers.batch_normalization(
                                embed_queries, training=training)
            query_features = tf.identity(query_features, name="query_features")

        s = tf.nn.l2_normalize(product_features, 1) * tf.nn.l2_normalize(query_features, 1)
        s = tf.reduce_sum(s, axis=1)
        s = tf.identity(s, name="score")

        return s

