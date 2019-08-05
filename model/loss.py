import tensorflow as tf

EPSILON_POS = 0.9
EPSILON_NEG = 0.2
EPSILON_ZERO = 0.55
POWER_M = 2


def semranker_loss(target_indices, predicted_scores):
    oh = tf.one_hot(target_indices, dtype=tf.float32, depth=3)

    l_pos = tf.expand_dims(tf.pow((-tf.minimum(0., predicted_scores - EPSILON_POS)), POWER_M), axis=1)
    l_zero = tf.expand_dims(tf.pow(tf.maximum(0., predicted_scores - EPSILON_ZERO), POWER_M), axis=1)
    l_neg = tf.expand_dims(tf.pow(tf.maximum(0., predicted_scores - EPSILON_NEG), POWER_M), axis=1)

    l = tf.concat([l_neg, l_zero, l_pos], axis=1) * oh

    loss = tf.reduce_sum(tf.reduce_mean(l, axis=0))

    return loss