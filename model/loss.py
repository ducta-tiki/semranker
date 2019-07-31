import tensorflow as tf

EPSILON_POS = 0.9
EPSILON_NEG = 0.2
EPSILON_ZERO = 0.55
POWER_M = 1.5


def semranker_loss(target_indices, predicted_scores):
    oh = tf.one_hot(target_indices, dtype=tf.float32)

    l_pos = tf.pow((-tf.minimum(0., predicted_scores - EPSILON_POS)), POWER_M)
    l_neg = tf.pow(tf.maximum(0., predicted_scores - EPSILON_NEG), POWER_M)
    l_zero = tf.pow(tf.maximum(0., predicted_scores - EPSILON_ZERO), POWER_M)

    l = tf.concat([l_zero, l_neg, l_pos]) * oh

    batch_size = tf.shape(target_indices)[0]
    loss = tf.reduce_sum(tf.reduce_mean(l, axis=0))

    return loss