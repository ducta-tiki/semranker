import tensorflow as tf
import pyhash


class CsvSemRankerReader(tf.contrib.data.CsvDataset):
    def __init__(
        self, vocab=list(),
        cat_tokens=list(), 
        attr_tokens=list(),
        maximum_product_name=[50, 50, 80], #for unigram, bigram, character trigrams
        maximum_cat=[10, 10, 20], #for unigram, bigram, character trigrams
        maximum_attr=[10, 10, 20], #for unigram, bigram, character trigrams
        unknown_bin=8012, *args, **kwargs):

        self.vocab = vocab
        self.cat_tokens = cat_tokens
        self.attr_tokens = attr_tokens
        self.unknown_bin = unknown_bin

        self.maximum_product_name = maximum_product_name
        self.maximum_cat = maximum_cat
        self.maximum_attr = maximum_attr
        
        self.word_2_idx = {}
        self.cat_token_2_idx = {}
        self.attr_token_2_idx = {}
        
        for i, w in enumerate(self.vocab):
            self.word_2_idx[w] = i
        for i, w in enumerate(self.cat_tokens):
            self.cat_token_2_idx[w] = i
        for i, w in enumerate(self.attr_tokens):
            self.attr_token_2_idx[w] = i

        self.hasher = pyhash.murmur3_32()
        super(CsvNerReader, self).__init__(*args, **kwargs)

    def unknown_to_idx(self, unknown):
        return self.hasher(unknown) % self.unknown_bin
    
    def _wrapper_map(self):
        def _map_to_indices(batch):
            pass
        return _map_to_indices

    def get_batch(self, batch_size, preprocessing_fn=None, epochs=None):
        pass

    def input_fn_generator(self, batch_size, preprocessing_fn=None, epochs=None):
        pass

