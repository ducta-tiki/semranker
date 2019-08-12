import tensorflow as tf
from model.loss import semranker_loss
from model.semranker import SemRanker
tf.logging.set_verbosity(tf.logging.INFO)


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
        loss = semranker_loss(labels, score)
    
    if is_training:
        global_step = tf.train.get_or_create_global_step()
        learning_rate = tf.train.exponential_decay(
            pconfig.get('init_learning_rate', 0.1),
            global_step,
            pconfig.get('step_change_learning_rate', 10000),
            pconfig.get('decay_learning_rate_ratio', 0.9),
            staircase=True)

        opt = tf.train.MomentumOptimizer(
            learning_rate=learning_rate,
            momentum=pconfig.get('momentum', 0.9)
        )
        #opt = tf.train.AdamOptimizer(
        #    learning_rate=learning_rate
        #)
        tensors_to_log = {'learning_rate': learning_rate, 'loss': loss}
        logging_hook = tf.train.LoggingTensorHook(
            tensors=tensors_to_log, every_n_iter=pconfig.get('step_print_logs', 5))
        training_hooks = [logging_hook]

        grads = opt.compute_gradients(loss)
        train_op = opt.apply_gradients(grads, global_step=global_step)
        update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
        train_op = tf.group([train_op, update_ops])
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