#!/usr/bin/env python
# _*_ coding:utf-8 _*_

from tensorflow.examples.tutorials.mnist import input_data
import tensorflow as tf

mnist = input_data.read_data_sets('../data/MNIST_data', one_hot=True)

sess = tf.InteractiveSession()

# 声明占位符
x = tf.placeholder(tf.float32, shape=[None, 784])
y_ = tf.placeholder(tf.float32, shape=[None, 10])


def weight_variables(shape):
    initial = tf.truncated_normal(shape=shape, stddev=0.1)
    return tf.Variable(initial)


def bias_variables(shape):
    initial = tf.constant(0.1, shape=shape)
    return tf.Variable(initial)


def conv2d(x, W):
    return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')


def max_pool_2x2(x):
    return tf.nn.max_pool(x, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')


x_image = tf.reshape(x, [-1, 28, 28, 1])

# 第一层卷积层
W_conv1 = weight_variables([5, 5, 1, 32])
b_conv1 = bias_variables([32])
h_conv1 = tf.nn.relu(conv2d(x_image, W_conv1) + b_conv1)
h_pool1 = max_pool_2x2(h_conv1)

# 第二层卷积层
W_conv2 = weight_variables([5, 5, 32, 64])
b_conv2 = bias_variables([64])
h_conv2 = tf.nn.relu(conv2d(h_pool1, W_conv2) + b_conv2)
h_pool2 = max_pool_2x2(h_conv2)

# 第一层全连接层
W_fc1 = weight_variables([7 * 7 * 64, 1024])
b_fc1 = bias_variables([1024])
h_pool2_flat = tf.reshape([h_pool2], [-1, 7 * 7 * 64])
h_fc1 = tf.nn.relu(tf.matmul(h_pool2_flat, W_fc1) + b_fc1)

# dropout层
keep_prob = tf.placeholder(tf.float32)
h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob)

# 第二层全连接层(最后输出层不要relu函数)
W_fc2 = weight_variables([1024, 10])
b_fc2 = bias_variables([10])
y = tf.matmul(h_fc1_drop, W_fc2) + b_fc2

# 定义损失函数和优化器
loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=y_, logits=y))
train_step = tf.train.AdamOptimizer(1e-4).minimize(loss)

correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(y_, 1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

with tf.Session() as sess:
    # 初始化变量
    init = tf.global_variables_initializer()
    sess.run(init)

    for i in range(10000):
        batch = mnist.train.next_batch(64)
        if i % 100 == 0:
            train_accuracy = accuracy.eval(feed_dict={
                x: batch[0], y_: batch[1], keep_prob: 1.0})
            print('step %d, training accuracy %g' % (i, train_accuracy))
        train_step.run(feed_dict={x: batch[0], y_: batch[1], keep_prob: 0.5})

    print('test accuracy1 %g' % accuracy.eval(feed_dict={
        x: mnist.test.images[:5000], y_: mnist.test.labels[:5000], keep_prob: 1.0
    }))

    print('test accuracy2 %g' % accuracy.eval(feed_dict={
        x: mnist.test.images[5000:], y_: mnist.test.labels[5000:], keep_prob: 1.0
    }))
