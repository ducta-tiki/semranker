import tensorflow as tf
from model.loss import semranker_loss
from model.semranker import SemRanker
from ndcg import ndcg
import numpy as np
tf.logging.set_verbosity(tf.logging.INFO)


def py_eval_func(target_scores, predict_scores, qids):
    target_scores = list(target_scores.numpy())    
    predict_scores = list(predict_scores.numpy())
    qids = list(qids.numpy())
    x = list(zip(target_scores, predict_scores, qids))

    groups = {}
    for e in x:
        if e[2] in groups:
            groups[e[2]].append((e[0], e[1]))
        else:
            groups[e[2]] = [(e[0], e[1])]
    
    cum_ndcg = []
    for _, g in groups.items():
        sorted_g = sorted(g, key=lambda x: x[1], reverse=True)
        pos = range(1, len(sorted_g)+1)
        rel = [e[0] for e in sorted_g]
        partial_ndcg = ndcg(pos, rel)
        if partial_ndcg > 0.:
            # print(partial_ndcg)
            cum_ndcg.append(partial_ndcg)
            
    return np.asarray(cum_ndcg, dtype=np.float32)


def semranker_fn(features, labels, mode, params):
    is_training = (mode == tf.estimator.ModeKeys.TRAIN)
    mconfig = params['model'] # model hyperparams
    pconfig = params.get('train') # process configuration (init learning rate, saving_dir,...)

    if not is_training:
        pconfig = params['eval']

   
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
    number_of_queries = features.get('number_of_queries')
    
    del features['number_of_queries']
    score = ranker(
        training=is_training,
        **features
    )
    param_stats = tf.profiler.profile(
        tf.get_default_graph(),
        options=tf.profiler.ProfileOptionBuilder
            .trainable_variables_parameter())
    print('Total params: %d\n' % param_stats.total_parameters)

    predictions = {
        'score': score,
    }

    if mode == tf.estimator.ModeKeys.PREDICT:
        return tf.estimator.EstimatorSpec(
            mode=mode,
            predictions=predictions,
            export_outputs={'predict_output': tf.estimator.export.PredictOutput(predictions)}
        )
    else:
        # loss = semranker_loss(labels["labels"], labels["weights"], score, 
        loss = semranker_loss(labels["labels"], None, score, 
            tf.cast(number_of_queries, tf.float32))
    
    if is_training:
        global_step = tf.train.get_or_create_global_step()
        opt = tf.train.AdamOptimizer()

        qids = features.get('qids')
        zndcg = tf.py_function(
                    py_eval_func, 
                    [labels["labels"], score, qids], 
                    [tf.float32])
        zndcg = tf.reshape(zndcg, [-1])
        ndcg_metrics = tf.reduce_mean(zndcg)
        tensors_to_log = {'loss': loss, 'ndcg': ndcg_metrics}
        logging_hook = tf.train.LoggingTensorHook(
            tensors=tensors_to_log, every_n_iter=pconfig.get('step_print_logs', 5))
        training_hooks = [logging_hook]

        grads = opt.compute_gradients(loss)
        train_op = opt.apply_gradients(grads, global_step=global_step)
        update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
        train_op = tf.group([train_op, update_ops])
        # train_op = tf.no_op()
        # train_op = tf.group([update_ops])
    else:
        train_op = None
        training_hooks = []

    return tf.estimator.EstimatorSpec(
        mode=mode,
        predictions=predictions,
        loss=loss,
        train_op=train_op,
        eval_metric_ops=None,
        training_hooks=training_hooks
    )