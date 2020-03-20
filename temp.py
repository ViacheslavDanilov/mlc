import tensorflow as tf

y_pred = tf.constant([0.85, 0.15, 0.70])
y_true = tf.constant([1.00, 0.00, 1.00])

# LOSS (differentiable)
y_true = tf.cast(y_true, tf.float32)
y_pred = tf.cast(y_pred, tf.float32)

tp_val = y_pred * y_true
fp_val = y_pred * (1 - y_true)
fn_val = (1 - y_pred) * y_true
tp = tf.reduce_sum(y_pred * y_true, axis=0)
fp = tf.reduce_sum(y_pred * (1 - y_true), axis=0)
fn = tf.reduce_sum((1 - y_pred) * y_true, axis=0)
soft_f1 = 2 * tp / (2 * tp + fn + fp + 1e-16)
cost = 1 - soft_f1  # reduce (1 - f1) in order to increase f1
macro_cost = tf.reduce_mean(cost)  # average on all labels

# F1 (not differentiable)
thresh = 0.5
y_pred = tf.cast(tf.greater(y_pred, thresh), tf.float32)
tp = tf.cast(tf.math.count_nonzero(y_pred * y_true, axis=0), tf.float32)
fp = tf.cast(tf.math.count_nonzero(y_pred * (1 - y_true), axis=0), tf.float32)
fn = tf.cast(tf.math.count_nonzero((1 - y_pred) * y_true, axis=0), tf.float32)
f1 = 2 * tp / (2 * tp + fn + fp + 1e-16)
macro_f1 = tf.reduce_mean(f1)