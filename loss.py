import keras.backend as k
import tensorflow as tf


def weighted_cross_entropy(y_true, y_pred):
    y_pred = y_pred[:, :, :, 1:7, :]
    y_true = y_true[:, :, :, 1:7, :]
    y_pred_f = k.reshape(y_pred, (-1, 3))
    y_true_f = k.reshape(y_true, (-1,))

    soft_pred_f = k.softmax(y_pred_f)
    soft_pred_f = k.log(tf.clip_by_value(soft_pred_f, 1e-10, 1.0))

    neg = k.equal(y_true_f, k.zeros_like(y_true_f))
    neg_loss = tf.gather(soft_pred_f[:, 0], tf.where(neg))

    pos1 = k.equal(y_true_f, k.ones_like(y_true_f))
    pos1_loss = tf.gather(soft_pred_f[:, 1], tf.where(pos1))

    pos2 = k.equal(y_true_f, 2 * k.ones_like(y_true_f))
    pos2_loss = tf.gather(soft_pred_f[:, 2], tf.where(pos2))

    # 0.78 = weight for background, 0.65 = liver, and 8.57 = tumor
    loss = -k.mean(tf.concat([0.78 * neg_loss, 0.65 * pos1_loss, 8.57 * pos2_loss], 0))

    return loss


def weighted_cross_entropy_2d(y_true, y_pred):

    y_pred_f = k.reshape(y_pred, (-1, 3))
    y_true_f = k.reshape(y_true, (-1,))

    soft_pred_f = k.softmax(y_pred_f)
    soft_pred_f = k.log(tf.clip_by_value(soft_pred_f, 1e-10, 1.0))

    neg = k.equal(y_true_f, k.zeros_like(y_true_f))
    neg_loss = tf.gather(soft_pred_f[:, 0], tf.where(neg))

    pos1 = k.equal(y_true_f, k.ones_like(y_true_f))
    pos1_loss = tf.gather(soft_pred_f[:, 1], tf.where(pos1))

    pos2 = k.equal(y_true_f, 2 * k.ones_like(y_true_f))
    pos2_loss = tf.gather(soft_pred_f[:, 2], tf.where(pos2))

    loss = -k.mean(tf.concat([0.78 * neg_loss, 0.65 * pos1_loss, 8.57 * pos2_loss], 0))

    return loss

