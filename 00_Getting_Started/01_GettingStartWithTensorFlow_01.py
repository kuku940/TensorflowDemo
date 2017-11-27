#!/usr/bin/env python
# _*_ coding:utf-8 _*_

import tensorflow as tf

# 申明Tensor
node1 = tf.constant(3.0, dtype=tf.float32)
node2 = tf.constant(4.0)  # also tf.float32 implicitly
print(node1, node2)

# 创建Sesssion对象
sess = tf.Session()
print(sess.run([node1, node2]))

# 使用和操作
node3 = tf.add(node1, node2)
print("node3:", node3)
print("sess.run(node3):", sess.run(node3))

# 使用占位符
a = tf.placeholder(tf.float32)
b = tf.placeholder(tf.float32)
adder_node = a + b

print(sess.run(adder_node, feed_dict={a: 3, b: 4.5}))
print(sess.run(adder_node, feed_dict={a: [1, 3], b: [2, 4]}))

add_add_triple = adder_node * 3
print(sess.run(add_add_triple, {a: 3, b: 4.5}))

# 创建线性模型
W = tf.Variable([.3], dtype=tf.float32)
b = tf.Variable([-.3], dtype=tf.float32)
x = tf.placeholder(tf.float32)
liear_model = W * x + b

# 初始化变量，常量在调用时被初始化
init = tf.global_variables_initializer()
sess.run(init)

print(sess.run(liear_model, {x: [1, 2, 3, 4]}))

# 定义label,loss
y = tf.placeholder(tf.float32)
squared_deltas = tf.square(liear_model - y)
loss = tf.reduce_sum(squared_deltas)
print(sess.run(loss, {x: [1, 2, 3, 4], y: [0, -1, -2, -3]}))

# 重新给W,b赋值，重新计算loss
fixW = tf.assign(W, [-1.])
fixb = tf.assign(b, [1.])
sess.run([fixW, fixb])
print(sess.run(loss, {x: [1, 2, 3, 4], y: [0, -1, -2, -3]}))

# 定义优化器
optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.01)
train = optimizer.minimize(loss)

# training data
x_train = [1, 2, 3, 4]
y_train = [0, -1, -2, -3]

# 训练模型train_loop
sess.run(init)
for i in range(1000):
    sess.run(train, {x: x_train, y: y_train})

# 计算训练集数据精度
curr_W, curr_b, curr_loss = sess.run([W, b, loss], {x: x_train, y: y_train})
print("W: %s b: %s loss: %s" % (curr_W, curr_b, curr_loss))
