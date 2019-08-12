import tensorflow as tf
from reader.tf_csv_reader import CsvSemRankerReader
from model.estimator import semranker_fn


def main():
    num_gpus = 8
    reader = CsvSemRankerReader(
        pair_path="shuf_pairs.csv",
        precomputed_path="meta/precomputed.json",
        product_db="db/tiki-products.db",
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
        'embed_size': 100,
        'attr_cat_embed_size': 10,
        'filter_sizes': [2,3,4,5],
        'max_query_length': reader.maximums_query[0],
        'max_product_name_length': reader.maximums_product_name[0],
        'max_brand_length': reader.maximums_brand[0],
        'max_author_length': reader.maximums_author[0],
        'max_cat_length': reader.maximums_cat[0],
        'max_attr_length': reader.maximums_attr[0],
        'num_filters': 20
    }

    pconfig = {
        'init_learning_rate': 0.1,
        'step_change_learning_rate': 100000,
        'decay_learning_rate_ratio': 0.1,
        'momentum': 0.9,
        'step_print_logs': 10,
        'batch_size': 1024,
        'max_steps': 4000000,
        'save_checkpoint_steps': 1000,
        'keep_checkpoint_max': 10
    }

    params = {'model': mconfig, 'train': pconfig, 'using_gpu': True}

    distribution = tf.contrib.distribute.MirroredStrategy(num_gpus=8)


    session_config = tf.ConfigProto(
        allow_soft_placement=True,
        operation_timeout_in_ms=10000,
        train_distribute=distribution)

    # build config for trainning
    run_config = tf.estimator.RunConfig(
        session_config=session_config,
        save_checkpoints_steps=pconfig['save_checkpoint_steps'],
        keep_checkpoint_max=pconfig['keep_checkpoint_max']
    )

    dnn_ranker = tf.estimator.Estimator(
        model_fn=semranker_fn,
        model_dir='checkpoint',
        config=run_config,
        params=params
    )

    dnn_ranker.train(
        input_fn=reader.input_fn_generator(pconfig['batch_size']),
        max_steps=pconfig['max_steps'])


if __name__ == "__main__":
    main()