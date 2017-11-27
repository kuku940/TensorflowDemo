#!/usr/bin/env python
# _*_ coding:utf-8 _*_

import tensorflow as tf


def cnn_model_fn(features, labels, mode):
    # Input Layer
    input_layer = tf.reshape(features["x"], [-1, 28, 28, 1])

    # 卷积层 [-1,28,28,32]
    conv1 = tf.layers.conv2d(inputs=input_layer, filters=32, kernel_size=[5, 5], padding='same', activation=tf.nn.relu)
    pool1 = tf.layers.max_pooling2d(inputs=conv1, pool_size=[2, 2], strides=2)  # [-1,14,14,32]

    conv2 = tf.layers.conv2d(pool1, 64, [5, 5], padding='same', activation=tf.nn.relu)  # [-1,14,14,64]
    pool2 = tf.layers.max_pooling2d(conv2, [2, 2], 2)  # [-1,7,7,64]

    # 全连接层
    pool2_flat = tf.reshape(pool2, [-1, 7, 7, 64])
    dense = tf.layers.dense(pool2_flat, 1024, tf.nn.relu)
    dropout = tf.layers.dropout(dense, rate=0.4, training=mode == tf.estimator.ModeKeys.TRAIN)

    # 预测层
    logits = tf.layers.dense(dropout, units=10)

    predictions = {
        "classes": tf.argmax(input=logits, axis=1),
        "probabilities": tf.nn.softmax(logits, name="softmax_tensor")
    }

    if mode == tf.estimator.ModeKeys.PREDICT:
        return tf.estimator.EstimatorSpec(mode=mode, predictions=predictions)

    # lass
    onehot_labels = tf.one_hot(indices=tf.cast(labels, tf.int32), depth=10)
    loss = tf.losses.softmax_cross_entropy(onehot_labels=onehot_labels, logits=logits)

    # 配置优化器和train_op
    if mode == tf.estimator.ModeKeys.TRAIN:
        optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.0001)
        train_op = optimizer.minimize(loss, global_step=tf.train.get_global_step())
        return tf.estimator.EstimatorSpec(mode=mode, loss=loss, train_op=train_op)

    #
    eval_metric_ops = {
        "accuracy":tf.metrics.accuracy(labels=labels,)
    }
