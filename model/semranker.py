import tensorflow as tf

# https://github.com/dennybritz/cnn-text-classification-tf/blob/master/text_cnn.py  


class SemRanker(object):
    def __init__(
        self, vocab_size, unknown_bin, cat_tokens_size, attr_tokens_size, 
        embed_size=80, filter_sizes=[2,3,4,5], 
        max_query_length=40, max_product_name_length=50,
        max_attr_length=20, max_cat_length=20, num_filters=5):
        self.embed_size = embed_size
        self.max_attr_length = max_attr_length
        self.max_cat_length = max_cat_length
        self.filter_sizes = filter_sizes
        self.num_filters = num_filters
        self.vocab_size = vocab_size
        self.unknown_bin = unknown_bin
        self.cat_tokens_size = cat_tokens_size
        self.attr_tokens_size = attr_tokens_size

    def text_cnn(self, embed_vecs, in_product, max_length, name):
        pooled_outputs = []
        for i, filter_size in enumerate(self.filter_sizes):
            with tf.name_scope("%s-conv-maxpool-%d" % (name, filter_size)):
                # Convolution
                filter_shape = [filter_size, self.embed_size, 1, self.num_filters]
                W = tf.Variable(tf.truncated_normal(filter_shape, stddev=0.1), name="W")
                b = tf.Variable(tf.constant(0.1, shape=[self.num_filters]), name="b")
                conv = tf.nn.conv2d(
                    embed_vecs,
                    W,
                    strides=[1, 1, 1, 1],
                    padding="VALID",
                    name="conv")
                # Apply nonlinearity
                h = tf.nn.relu(tf.nn.bias_add(conv, b), name="relu")
                # Maxpooling over the outputs
                pooled = tf.nn.max_pool(
                    h,
                    ksize=[1, max_length - filter_size + 1, 1, 1],
                    strides=[1, 1, 1, 1],
                    padding='VALID',
                    name="pool")
                pooled_outputs.append(pooled)
        num_filters_total = num_filters * len(filter_sizes)
        h_pool = tf.concat(pooled_outputs, 3)
        h_pool_flat = tf.reshape(h_pool, [-1, num_filters_total]) # sum_of(in_product)
        if in_product:
            rz = tf.reduce_sum(in_product)
            z1 = tf.cumsum(in_product, exclusive=True)
            z2 = tf.cumsum(in_product)
            mask = tf.sequence_mask(z2, maxlen=rz, dtype=tf.int32) - \
                tf.sequence_mask(z1, maxlen=rz, dtype=tf.int32)

            p = tf.matmul(mask, h_pool_flat)
            p = p / tf.cast(in_product, tf.float32) # batch_size x num_filters_total
            return p
        return h_pool_flat

    def cats_text_cnn(self, cats_embed, cats_in_product, max_len):
        return text_cnn(cats_embed, cats_in_product, max_len, 'cat')

    def attrs_text_cnn(self, attrs_embed, attrs_in_product, max_len):
        return text_cnn(attrs_embed, attrs_in_product, max_len, 'attr')

    def query_text_cnn(self, queries_embed, max_len):
        return text_cnn(queries_embed, None, max_len, 'query')

    def product_text_cnn(self, products_embed, max_len):
        return text_cnn(products_embed, None, max_len, 'product')

    def brand_text_cnn(self, brands_embed, max_len):
        return text_cnn(brands_embed, None, max_len, 'brand')

    def author_text_cnn(self, authors_embed, max_len):
        return text_cnn(authors_embed, None, max_len, 'author')

    def __call__(
        self, query_indices=None, product_name_indices=None, brand_indices=None, author_indices=None,
        cat_indices=None, attr_indices=None, cat_tokens=None, attr_tokens=None, 
        cats_in_product=None, attrs_in_product=None):
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
        :return: cosine score
        """

        with tf.variable_scope("embedding"):
            ngram_embed_weights = tf.get_variable(
                'n_gram_embedding', [self.vocab_size+self.unknown_bin, self.embed_size],
                initializer=tf.random_normal_initializer(mean=0, stddev=0.01)
            )
            zero_vector = tf.zeros([1, self.embed_size], dtype=tf.float32)
            ngram_embed_weights = tf.concat(
                [ngram_embed_weights, zero_vector], axis=0, name="zero_padding_n_gram_embedding")

            cat_embed_weights = tf.get_variable(
                'cat_embedding', [self.cat_tokens_size, self.embed_size]
            )
            cat_embed_weights = tf.concat(
                [cat_embed_weights, zero_vector], axis=0, name="zero_padding_cat_embedding")

            attr_embed_weights = tf.get_variable(
                'attr_embedding', [self.attr_tokens_size, self.embed_size]
            )
            attr_embed_weights = tf.concat(
                [attr_embed_weights, zero_vector], axis=0, name="zero_padding_attr_embedding")

        # with tf.name_scope("calc_embedding_feature"):
        #     pass
