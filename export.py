import tensorflow as tf
import argparse
from reader.tf_csv_reader import CsvSemRankerReader
from model.semranker import SemRanker

QUERY_UNIGRAM_INDICES = 'query_unigram_indices'
QUERY_BIGRAM_INDICES = 'query_bigram_indices'
QUERY_CHAR_TRIGRAM_INDICES = 'query_char_trigram_indices'
PRODUCT_UNIGRAM_INDICES = 'product_unigram_indices'
PRODUCT_BIGRAM_INDICES = 'product_bigram_indices'
PRODUCT_CHAR_TRIGRAM_INDICES = 'product_char_trigram_indices'
BRAND_UNIGRAM_INDICES = 'brand_unigram_indices'
BRAND_BIGRAM_INDICES = 'brand_bigram_indices'
BRAND_CHAR_TRIGRAM_INDICES = 'brand_char_trigram_indices'
AUTHOR_UNIGRAM_INDICES = 'author_unigram_indices'
AUTHOR_BIGRAM_INDICES = 'author_bigram_indices'
AUTHOR_CHAR_TRIGRAM_INDICES = 'author_char_trigram_indices'
CAT_UNIGRAM_INDICES = 'cat_unigram_indices'
CAT_BIGRAM_INDICES = 'cat_bigram_indices'
CAT_CHAR_TRIGRAM_INDICES = 'cat_char_trigram_indices'
CAT_TOKENS = 'cat_tokens'
CATS_IN_PRODUCT = 'cats_in_product'
ATTR_UNIGRAM_INDICES = 'attr_unigram_indices'
ATTR_BIGRAM_INDICES = 'attr_bigram_indices'
ATTR_CHAR_TRIGRAM_INDICES = 'attr_char_trigram_indices'
ATTR_TOKENS = 'attr_tokens'
ATTRS_IN_PRODUCT = 'attrs_in_product'
FREE_FEATURES = 'free_features'

SCORE = 'score:0'


def main():
    import os
    list_files = [os.path.join("transform_impressions", f) for f in os.listdir("transform_impressions") if f.endswith(".csv")]
    reader = CsvSemRankerReader(
        pair_paths=list_files,
        precomputed_path="meta/precomputed.json",
        product_db=None,
        vocab_path="meta/vocab.txt",
        cat_tokens_path="meta/cats.txt",
        attr_tokens_path="meta/attrs.txt",
        maximums_query=[25, 25, 125],#for unigram, bigram, character trigrams
        maximums_product_name=[50, 50, 250], #for unigram, bigram, character trigrams
        maximums_brand=[10, 10, 50],
        maximums_author=[10, 10, 50],
        maximums_cat=[10, 10, 50], #for unigram, bigram, character trigrams
        maximums_attr=[10, 10, 50], #for unigram, bigram, character trigrams
    )

    mconfig = {
        'vocab_size': reader.vocab_size,
        'unknown_bin': reader.unknown_bin,
        'cat_tokens_size': reader.cat_tokens_size,
        'attr_tokens_size': reader.attr_tokens_size,
        'embed_size': 128,
        'attr_cat_embed_size': 30,
        'filter_sizes': [2,3,4,5],
        'max_query_length': reader.maximums_query[0],
        'max_product_name_length': reader.maximums_product_name[0],
        'max_brand_length': reader.maximums_brand[0],
        'max_author_length': reader.maximums_author[0],
        'max_cat_length': reader.maximums_cat[0],
        'max_attr_length': reader.maximums_attr[0],
        'num_filters': 20
    }

    receiver_tensors = {
        'query_unigram_indices' : tf.placeholder(
            tf.int32, shape=[None, mconfig['max_query_length']], name=QUERY_UNIGRAM_INDICES),
        'query_bigram_indices': tf.placeholder(
            tf.int32, shape=[None, mconfig['max_query_length']], name=QUERY_BIGRAM_INDICES),
        'query_char_trigram_indices': tf.placeholder(
            tf.int32, shape=[None, mconfig['max_query_length']*5], name=QUERY_CHAR_TRIGRAM_INDICES),
        'product_unigram_indices': tf.placeholder(
            tf.int32, shape=[None, mconfig['max_product_name_length']], name=PRODUCT_UNIGRAM_INDICES),
        'product_bigram_indices': tf.placeholder(
            tf.int32, shape=[None, mconfig['max_product_name_length']], name=PRODUCT_BIGRAM_INDICES),
        'product_char_trigram_indices': tf.placeholder(
            tf.int32, shape=[None, mconfig['max_product_name_length']*5], name=PRODUCT_CHAR_TRIGRAM_INDICES),
        'brand_unigram_indices': tf.placeholder(
            tf.int32, shape=[None, mconfig['max_brand_length']], name=BRAND_UNIGRAM_INDICES),
        'brand_bigram_indices': tf.placeholder(
            tf.int32, shape=[None, mconfig['max_brand_length']], name=BRAND_BIGRAM_INDICES),
        'brand_char_trigram_indices': tf.placeholder(
            tf.int32, shape=[None, mconfig['max_brand_length']*5], name=BRAND_CHAR_TRIGRAM_INDICES),
        'author_unigram_indices': tf.placeholder(
            tf.int32, shape=[None, mconfig['max_author_length']], name=AUTHOR_UNIGRAM_INDICES),
        'author_bigram_indices': tf.placeholder(
            tf.int32, shape=[None, mconfig['max_author_length']], name=AUTHOR_BIGRAM_INDICES),
        'author_char_trigram_indices': tf.placeholder(
            tf.int32, shape=[None, mconfig['max_author_length']*5], name=AUTHOR_CHAR_TRIGRAM_INDICES),
        'cat_unigram_indices': tf.placeholder(
            tf.int32, shape=[None, mconfig['max_cat_length']], name=CAT_UNIGRAM_INDICES),
        'cat_bigram_indices': tf.placeholder(
        tf.int32, shape=[None, mconfig['max_cat_length']], name=CAT_BIGRAM_INDICES),
        'cat_char_trigram_indices': tf.placeholder(
            tf.int32, shape=[None, mconfig['max_cat_length']*5], name=CAT_CHAR_TRIGRAM_INDICES),
        'cat_tokens': tf.placeholder(tf.int32, shape=[None,], name=CAT_TOKENS),
        'cat_in_product': tf.placeholder(tf.int32, shape=[None,], name=CATS_IN_PRODUCT),
        'attr_unigram_indices': tf.placeholder(
            tf.int32, shape=[None, mconfig['max_attr_length']], name=ATTR_UNIGRAM_INDICES),
        'attr_bigram_indices': tf.placeholder(
            tf.int32, shape=[None, mconfig['max_attr_length']], name=ATTR_BIGRAM_INDICES),
        'attr_char_trigram_indices': tf.placeholder(
            tf.int32, shape=[None, mconfig['max_attr_length']*5], name=ATTR_CHAR_TRIGRAM_INDICES),
        'attr_tokens': tf.placeholder(tf.int32, shape=[None,], name=ATTR_TOKENS),
        'attr_in_product': tf.placeholder(tf.int32, shape=[None,], name=ATTRS_IN_PRODUCT),
        'features': tf.placeholder(tf.float32, shape=[None, len(reader.precomputed)], name=FREE_FEATURES),
        'training': False
    }

    ranker = SemRanker(
        vocab_size=mconfig.get('vocab_size'),
        unknown_bin=mconfig.get('unknown_bin'),
        cat_tokens_size=mconfig.get('cat_tokens_size'),
        attr_tokens_size=mconfig.get('attr_tokens_size'),
        embed_size=mconfig.get('embed_size'),
        attr_cat_embed_size=mconfig.get('attr_cat_embed_size'),
        filter_sizes=mconfig.get('filter_sizes'),
        max_query_length=mconfig.get('max_query_length'),
        max_product_name_length=mconfig.get('max_product_name_length'),
        max_brand_length=mconfig.get('max_brand_length'),
        max_author_length=mconfig.get('max_author_length'),
        max_attr_length=mconfig.get('max_attr_length'),
        max_cat_length=mconfig.get('max_cat_length'), 
        num_filters=mconfig.get('num_filters')
    )

    score = ranker(
        **receiver_tensors 
    )

    import shutil
    import os
    if os.path.exists("export"):
        shutil.rmtree("export")

    parser = argparse.ArgumentParser()
    parser.add_argument("--model", help="Path to config file", type=str,
                        default="30000")
    args = parser.parse_args()
    
    saver = tf.train.Saver()
    with tf.Session() as sess:
        saver.restore(sess, "checkpoint/model.ckpt-%s" % args.model)

        del receiver_tensors['training']
        tf.saved_model.simple_save(
            sess, "export", 
            inputs=receiver_tensors, 
            outputs={'score': score})


if __name__ == "__main__":
    main()