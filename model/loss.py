import tensorflow as tf


def HeSho(y_preds, y_true):
    #  losses = [tf.reduce_mean((y_true - y_pred)**2) for y_pred in y_preds]
    losses = tf.reduce_mean((y_true - y_preds)**2)
    loss = tf.reduce_mean(tf.stack(losses))
    return loss
