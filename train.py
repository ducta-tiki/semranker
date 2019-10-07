import tensorflow as tf
from reader.tf_csv_reader import CsvSemRankerReader
from reader.fast_tf_csv_reader import ProducerManager
from model.estimator import semranker_fn
import os

def main():    
    list_files = [
        os.path.join("transform_impressions", f) 
            for f in os.listdir("transform_impressions") if f.endswith(".csv")]
    # reader = CsvSemRankerReader(
    #     pair_paths=list_files,
    #     precomputed_path="meta/precomputed.json",
    #     product_db="data/product.csv",
    #     vocab_path="meta/vocab.txt",
    #     cat_tokens_path="meta/cats.txt",
    #     attr_tokens_path="meta/attrs.txt",
    #     maximums_query=[25, 25, 125],#for unigram, bigram, character trigrams
    #     maximums_product_name=[50, 50, 250], #for unigram, bigram, character trigrams
    #     maximums_brand=[10, 10, 50],
    #     maximums_author=[10, 10, 50],
    #     maximums_cat=[10, 10, 50], #for unigram, bigram, character trigrams
    #     maximums_attr=[10, 10, 50], #for unigram, bigram, character trigrams
    # )

    pconfig = {
        'init_learning_rate': 0.01,
        'step_change_learning_rate': 100000,
        'decay_learning_rate_ratio': 0.9,
        'momentum': 0.9,
        'step_print_logs': 10,
        'batch_size': 156,
        'max_steps': 4000000,
        'save_checkpoint_steps': 2000,
        'keep_checkpoint_max': 20
    }

    reader = ProducerManager(list_files, 
        precomputed_path="meta/precomputed.json",
        product_db= "db/precomputed-products.db",
        vocab_path="meta/vocab.txt",
        cat_tokens_path="meta/cats.txt",
        attr_tokens_path="meta/attrs.txt",
        maximums_query=[25, 25, 125],
        maximums_product_name=[50, 50, 250],
        maximums_brand=[10, 10, 50],
        maximums_author=[10, 10, 50],
        maximums_cat=[10, 10, 20],
        maximums_attr=[10, 10, 20],
        n_workers=8, limit_sample=100000000, batch_size=pconfig['batch_size'], warmup=20)

    mconfig = {
        'vocab_size': reader.vocab_size,
        'unknown_bin': reader.unknown_bin,
        'cat_tokens_size': reader.cat_tokens_size,
        'attr_tokens_size': reader.attr_tokens_size,
        'embed_size': 128,
        'attr_cat_embed_size': 64,
        'filter_sizes': [2,3,4,5],
        'max_query_length': reader.maximums_query[0],
        'max_product_name_length': reader.maximums_product_name[0],
        'max_brand_length': reader.maximums_brand[0],
        'max_author_length': reader.maximums_author[0],
        'max_cat_length': reader.maximums_cat[0],
        'max_attr_length': reader.maximums_attr[0],
        'num_filters': 20
    }


    params = {'model': mconfig, 'train': pconfig, 'using_gpu': True}

    using_gpu = params.get('using_gpu', False)

    device_fn = lambda op: "/cpu:0"
    if using_gpu:
        device_fn = lambda op: "/gpu:0"

    session_config = tf.ConfigProto(
        allow_soft_placement=True)

    # build config for trainning
    run_config = tf.estimator.RunConfig(
        device_fn=device_fn,
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
        input_fn=reader.input_fn_generator(),
        # input_fn=reader.input_fn_generator(pconfig.get('batch_size')),
        max_steps=pconfig['max_steps'])


if __name__ == "__main__":
    main()