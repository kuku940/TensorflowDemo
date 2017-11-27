#!/usr/bin/env python
# _*_ coding:utf-8 _*_

from tensorflow.examples.tutorials.mnist import input_data
import tensorflow as tf

mnist = input_data.read_data_sets("../data/MNIST_data/", one_hot=True)

x = tf.placeholder(tf.float32, [None, 784])

W = tf.Variable(tf.zeros([784, 10]))
b = tf.Variable(tf.zeros([10]))

y = tf.nn.softmax(tf.matmul(x, W) + b)

y_ = tf.placeholder(tf.float32, [None, 10])

# 定义交叉熵
# cross_entropy = tf.reduce_sum(y_ * tf.log(y), reduction_indices=[1])
cross_entropy = tf.nn.softmax_cross_entropy_with_logits(logits=y, labels=y_)

optimizer = tf.train.GradientDescentOptimizer(0.1)
train_step = optimizer.minimize(cross_entropy)

sess = tf.InteractiveSession()
init = tf.global_variables_initializer()
sess.run(init)

for _ in range(10000):
    batch_xs, batch_ys = mnist.train.next_batch(128)
    sess.run(train_step, feed_dict={x: batch_xs, y_: batch_ys})

correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(y_, 1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

print(sess.run(accuracy, feed_dict={x: mnist.test.images, y_: mnist.test.labels}))
# 0.658