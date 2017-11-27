#!/usr/bin/env python
# _*_ coding: utf-8 _*_

import tensorflow as tf
import numpy as np


def save():
    ## 保存到文件

    W = tf.Variable([[1, 2, 3], [4, 5, 6]], dtype=tf.float32, name='weights')
    b = tf.Variable([[1, 2, 3]], dtype=tf.float32, name='biases')

    init = tf.initialize_all_variables()

    saver = tf.train.Saver()

    with tf.Session() as sess:
        sess.run(init)
        save_path = saver.save(sess, '../data/SaveAndRestore/save_net.ckpt')
        print('Save path:', save_path)


def restore():
    ## 读取变量
    W1 = tf.Variable(np.arange(6).reshape((2, 3)), dtype=tf.float32, name='weights')
    b1 = tf.Variable(np.arange(3).reshape((1, 3)), dtype=tf.float32, name='biases')

    saver = tf.train.Saver()
    with tf.Session() as sess:
        saver.restore(sess, "../data/SaveAndRestore/save_net.ckpt")
        print('weights:', sess.run(W1))
        print('biases:', sess.run(b1))


if __name__ == "__main__":
    # save()
    restore()
