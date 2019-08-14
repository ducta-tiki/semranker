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

    def text_cnn(self, embed_vecs, in_product, max_length, name, tokens_embed=None):
        pooled_outputs = []
        for i, filter_size in enumerate(self.filter_sizes):
            with tf.name_scope("%s-conv-maxpool-%d" % (name, filter_size)):
                # Convolution
                filter_shape = [filter_size, self.embed_size, 1, self.num_filters]
                W = tf.Variable(tf.truncated_normal(filter_shape, stddev=0.1), name="W")
                b = tf.Variable(tf.constant(0.1, shape=[self.num_filters]), name="b")
                conv = tf.nn.conv2d(
                    tf.expand_dims(embed_vecs, -1),
                    W,
                    strides=[1, 1, 1, 1],
                    padding="VALID",
                    name="conv")
                # Apply nonlinearity
                # h = tf.nn.relu(tf.nn.bias_add(conv, b), name="relu")
                # Maxpooling over the outputs
                pooled = tf.nn.avg_pool(
                    tf.nn.bias_add(conv, b),
                    ksize=[1, max_length - filter_size + 1, 1, 1],
                    strides=[1, 1, 1, 1],
                    padding='VALID',
                    name="pool")
                pooled_outputs.append(pooled)
        num_filters_total = self.num_filters * len(self.filter_sizes)
        h_pool = tf.concat(pooled_outputs, 3)
        h_pool_flat = tf.reshape(h_pool, [-1, num_filters_total]) # sum_of(in_product) x num_filters_total

        if not in_product is None:
            h_pool_flat = tf.concat([h_pool_flat, tokens_embed], axis=1) # sum_of(in_product) x (num_filters_total + token_embed_size)
            rz = tf.reduce_sum(in_product)
            z1 = tf.cumsum(in_product, exclusive=True)
            z2 = tf.cumsum(in_product)
            mask = tf.sequence_mask(z2, maxlen=rz, dtype=tf.float32) - \
                tf.sequence_mask(z1, maxlen=rz, dtype=tf.float32) # batch_size x sum_of(in_product)
            p = tf.matmul(mask, h_pool_flat)
            p = p / tf.expand_dims(tf.cast(in_product, tf.float32), axis=1) # batch_size x (num_filters_total + token_embed_size)
            return p
        return h_pool_flat

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
            for qz, name in zip(query_indices, ['unigram', 'bigram', 'char_trigram']):
                qz_embed = tf.nn.embedding_lookup(ngram_embed_weights, qz)

                embed_queries.append(
                    tf.reduce_mean(tf.transpose(qz_embed, [0, 2, 1]), axis=2))

            embed_queries = (embed_queries[0] + embed_queries[1] + embed_queries[2]) / 3.
            embed_queries = tf.identity(embed_queries, name="query_encode")

            embed_product_names = []
            for pz, name in zip(product_name_indices, ['unigram', 'bigram', 'char_trigram']):
                pz_embed = tf.nn.embedding_lookup(ngram_embed_weights, pz)
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
                units=self.embed_size,
                kernel_initializer=tf.random_normal_initializer(mean=0, stddev=0.01),
                activation=tf.nn.tanh
            )
            product_features = tf.layers.batch_normalization(
                                v1, training=training)
            product_features = tf.identity(product_features, name="product_encode")

            query_features = tf.layers.batch_normalization(
                                embed_queries, training=training)
            query_features = tf.nn.tanh(query_features)
            query_features = tf.identity(query_features, name="query_features")

        s = tf.nn.l2_normalize(product_features, 1) * tf.nn.l2_normalize(query_features, 1)
        s = tf.reduce_sum(s, axis=1)
        s = tf.identity(s, name="score")

        return s

    def backup__call__(
        self, query_unigram_indices=None, query_bigram_indices=None, query_char_trigram_indices=None,
        product_unigram_indices=None, product_bigram_indices=None, product_char_trigram_indices=None,
        brand_unigram_indices=None, brand_bigram_indices=None, brand_char_trigram_indices=None,
        author_unigram_indices=None, author_bigram_indices=None, author_char_trigram_indices=None,
        cat_tokens=None, cat_in_product=None, cat_unigram_indices=None, cat_bigram_indices=None, 
        cat_char_trigram_indices=None, attr_tokens=None, attr_in_product=None, attr_unigram_indices=None,
        attr_bigram_indices=None, attr_char_trigram_indices=None, features=None, qids=None, training=True):
        """
        :param query_indices: (unigrams, bigrams, char_trigrams) of query
                    [(batch_size, max_len_1), (batch_size, max_len_2), (batch_size, max_len_3)]
        :param product_name_indices: (unigrams, bigrams, char_trigrams) of product_name
                    [(batch_size, max_len_1), (batch_size, max_len_2), (batch_size, max_len_3)]
        :param brand_indices: (unigrams, bigrams, char_trigrams) of brand
                    [(batch_size, max_len_1), (batch_size, max_len_2), (batch_size, max_len_3)]
        :param author_indices: (unigrams, bigrams, char_trigrams) of author
                    [(batch_size, max_len_1), (batch_size, max_len_2), (batch_size, max_len_3)]
        :param cat_indices: (unigrams, bigrams, char_trigrams) of category
                    [(sum_of_cat_tokens, max_len_1), (sum_of_cat_tokens, max_len_2), (sum_of_cat_tokens, max_len_3)]
        :param attr_indices: (unigrams, bigrams, char_trigrams) of attributes
                    [(sum_of_attr_tokens, max_len_1), (sum_of_attr_tokens, max_len_2), (sum_of_attr_tokens, max_len_3)]
        :param cat_tokens: categories tokens of product
                    (sum_of_cat_tokens,)
        :param attr_tokens: attributes tokens of product
                    (sum_of_attr_tokens,)
        :param cats_in_product: How many categories in a product
                    (batch_size,)
        :param attrs_in_product: How many attributes in a product
                    (batch_size,)
        :param free_features: Free features of products
                    (batch_size, number_of_features)
        :return: cosine score
        """

        query_indices = [query_unigram_indices, query_bigram_indices, query_char_trigram_indices]
        product_name_indices = [product_unigram_indices, product_bigram_indices, product_char_trigram_indices]
        brand_indices = [brand_unigram_indices, brand_bigram_indices, brand_char_trigram_indices]
        author_indices = [author_unigram_indices, author_bigram_indices, author_char_trigram_indices]
        cat_indices = [cat_unigram_indices, cat_bigram_indices, cat_char_trigram_indices]
        attr_indices = [ attr_unigram_indices, attr_bigram_indices, attr_char_trigram_indices]

        with tf.variable_scope("embedding"):
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
            for qz, name, scale in zip(query_indices, ['unigram', 'bigram', 'char_trigram'], [1,1,5]):
                qz_embed = tf.nn.embedding_lookup(ngram_embed_weights, qz)
                qz_cnn = self.text_cnn(qz_embed, None, self.max_query_length*scale, "query-" + name)
                embed_queries.append(qz_cnn)

            embed_queries = tf.concat(embed_queries, axis=1, name="query_encode")

            embed_product_names = []
            for pz, name, scale in zip(product_name_indices, ['unigram', 'bigram', 'char_trigram'], [1,1,5]):
                pz_embed = tf.nn.embedding_lookup(ngram_embed_weights, pz)
                pz_cnn = self.text_cnn(pz_embed, None, self.max_product_name_length*scale, "product-" + name)
                embed_product_names.append(pz_cnn)

            embed_product_names = tf.concat(embed_product_names, axis=1, name="product_name_encode")

            embed_brand = []
            for bz, name, scale in zip(brand_indices, ['unigram', 'bigram', 'char_trigram'], [1,1,5]):
                bz_embed = tf.nn.embedding_lookup(ngram_embed_weights, bz)
                bz_cnn = self.text_cnn(bz_embed, None, self.max_brand_length*scale, "brand-" + name)
                embed_brand.append(bz_cnn)

            embed_brand = tf.concat(embed_brand, axis=1, name="brand_encode")

            embed_author = []
            for az, name, scale in zip(author_indices, ['unigram', 'bigram', 'char_trigram'], [1,1,5]):
                az_embed = tf.nn.embedding_lookup(ngram_embed_weights, az)
                az_cnn = self.text_cnn(az_embed, None, self.max_author_length*scale, "author-" + name)
                embed_author.append(az_cnn)

            embed_author = tf.concat(embed_author, axis=1, name="author_encode")

            embed_cat_tokens = tf.nn.embedding_lookup(cat_embed_weights, cat_tokens)
            embed_cat = []
            for cz, name, scale in zip(cat_indices, ['unigram', 'bigram', 'char_trigram'], [1,1,5]):
                cz_embed = tf.nn.embedding_lookup(ngram_embed_weights, cz)
                cz_cnn = self.text_cnn(
                    cz_embed, cat_in_product, self.max_cat_length*scale, "category-" + name, embed_cat_tokens)
                embed_cat.append(cz_cnn)
            embed_cat = tf.concat(embed_cat, axis=1, name="category_encode")

            embed_attr_tokens = tf.nn.embedding_lookup(attr_embed_weights, attr_tokens)
            embed_attr = []
            for attr_z, name, scale in zip(attr_indices, ['unigram', 'bigram', 'char_trigram'], [1,1,5]):
                attr_z_embed = tf.nn.embedding_lookup(ngram_embed_weights, attr_z)
                attr_z_cnn = self.text_cnn(
                    attr_z_embed, attr_in_product, self.max_cat_length*scale, "attribute-" + name, embed_attr_tokens)
                embed_attr.append(attr_z_cnn)
            embed_attr = tf.concat(embed_attr, axis=1, name="attribute_encode")

            embed_cat = tf.reshape(embed_cat, 
                [-1, (len(self.filter_sizes)*self.num_filters + self.attr_cat_embed_size)*3])
            embed_attr= tf.reshape(embed_cat, 
                [-1, (len(self.filter_sizes)*self.num_filters + self.attr_cat_embed_size)*3])
            product_features = tf.concat([
                embed_product_names, embed_brand, embed_author, embed_cat, embed_attr, features
            ], axis=1)

        with tf.name_scope("bn"):
            product_features = tf.layers.batch_normalization(
                                product_features, training=training)
            product_features = tf.identity(product_features, name="product_encode")

            query_features = tf.layers.batch_normalization(
                                embed_queries, training=training)
            query_features = tf.nn.tanh(query_features)
            query_features = tf.identity(query_features, name="query_features")

        with tf.variable_scope("dense"):
            v1 = tf.layers.dense(
                product_features,
                units=self.num_filters * len(self.filter_sizes) * 3,
                kernel_initializer=tf.random_normal_initializer(mean=0, stddev=0.01),
                activation=tf.nn.tanh
            )
        s = tf.nn.l2_normalize(v1, 1) * tf.nn.l2_normalize(query_features, 1)
        s = tf.reduce_sum(s, axis=1)
        s = tf.identity(s, name="score")

        return s