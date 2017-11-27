#!/usr/bin/env python
# _*_ coding:utf-8 _*_

import tensorflow as tf
import numpy as np

tf.set_random_seed(1)
np.random.seed(1)

# fake data
x = np.linspace(-1, 1, 100)[:, np.newaxis]
noise = np.random.normal(0, 0.1, size=x.shape)
y = np.power(x, 2) + noise

with tf.variable_scope('Inputs'):
    tf_x = tf.placeholder(tf.float32, x.shape, name='x')
    tf_y = tf.placeholder(tf.float32, y.shape, name='y')

with tf.variable_scope('Net'):
    l1 = tf.layers.dense(tf_x, 10, tf.nn.relu, name='hidden_layer')
    output = tf.layers.dense(l1, 1, name='output_layer')

    # add to histogram summary - 添加信息到Tensor图中
    tf.summary.histogram('h_out', l1)
    tf.summary.histogram('pred', output)

loss = tf.losses.mean_squared_error(tf_y, output, scope='loss')
train_op = tf.train.GradientDescentOptimizer(learning_rate=0.5).minimize(loss)
tf.summary.scalar('loss', loss)  # add loss to scalar summary - 添加标量统计结果

sess = tf.Session()
sess.run(tf.global_variables_initializer())

# 合并summary数据，并将他们写入磁盘
merge_op = tf.summary.merge_all()
writer = tf.summary.FileWriter('../data/tensorboard/logs', sess.graph)

for step in range(1000):
    # train and net output
    _, summary = sess.run([train_op, merge_op], feed_dict={tf_x: x, tf_y: y})
    writer.add_summary(summary, step)

# Lastly, in your terminal or CMD, type this :
# $ tensorboard --logdir path/to/log
# open you google chrome, type the link shown on your terminal or CMD. (something like this: http://localhost:6006)
