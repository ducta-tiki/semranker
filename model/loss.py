import tensorflow as tf

EPSILON_POS = 0.9
EPSILON_NEG = 0.2
EPSILON_ZERO_H = 0.55
EPSILON_ZERO_L = 0.25
POWER_M = 2


def semranker_loss(target_indices, weights, predicted_scores, batch_size=100):
    oh = tf.one_hot(target_indices, dtype=tf.float32, depth=3)

    l_pos = tf.expand_dims(tf.pow(-tf.minimum(0., predicted_scores - EPSILON_POS), POWER_M), axis=1)
    
    l_zero = tf.expand_dims(tf.pow(tf.maximum(0., predicted_scores - EPSILON_ZERO_H), POWER_M), axis=1) + \
            tf.expand_dims(tf.pow(-tf.minimum(0., predicted_scores - EPSILON_ZERO_L), POWER_M), axis=1)

    l_neg = tf.expand_dims(tf.pow(tf.maximum(0., predicted_scores - EPSILON_NEG), POWER_M), axis=1)

    l = tf.concat([l_neg, l_zero, l_pos], axis=1) * oh

    #l = tf.math.multiply(l, tf.expand_dims(weights, -1))

    loss = tf.reduce_sum(l)/batch_size

    return loss