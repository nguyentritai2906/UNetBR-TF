import tensorflow as tf


def HeSho(y_preds, y_true):
    loss = tf.reduce_mean((y_true - y_preds)**2)
    return loss
