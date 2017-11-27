#!/usr/bin/env python
# _*_ coding:utf-8 _*_

import tensorflow as tf
import numpy as np

dataset1 = tf.data.Dataset.from_tensor_slices(tf.random_uniform([4, 10]))
print(dataset1.output_types)
print(dataset1.output_shapes)

dataset2 = tf.data.Dataset.from_tensor_slices(
    (tf.random_uniform([4]), tf.random_uniform([4, 100], maxval=100, dtype=tf.int32)))
print(dataset2.output_types)
print(dataset2.output_shapes)

dataset3 = tf.data.Dataset.zip((dataset1, dataset2))
print(dataset3.output_types)
print(dataset3.output_shapes)

dataset = tf.data.Dataset.from_tensor_slices(
    {"a": tf.random_uniform([4]), "b": tf.random_uniform([4, 100], maxval=100, dtype=tf.int32)})
print(dataset.output_types)
print(dataset.output_shapes)

## 创建一个迭代器来获取元素
dataset = tf.data.Dataset.range(100)
iterator = dataset.make_one_shot_iterator()
next_element = iterator.get_next()

sess = tf.Session()
for i in range(100):
    value = sess.run(next_element)
    assert i == value

##
max_value = tf.placeholder(tf.int64, shape=[])
dataset = tf.data.Dataset.range(max_value)
iterator = dataset.make_initializable_iterator()
next_element = iterator.get_next()

sess.run(iterator.initializer, feed_dict={max_value: 10})
for i in range(10):
    value = sess.run(next_element)
    assert i == value

sess.run(iterator.initializer, feed_dict={max_value: 10})
for i in range(10):
    value = sess.run(next_element)
    assert i == value

# 定义
training_dataset = tf.data.Dataset.range(100).map(
    lambda x: x + tf.random_uniform([], -10, 10, tf.int64))
validation_dataset = tf.data.Dataset.range(50)

Iterator = tf.contrib.data.Iterator
iterator = Iterator.from_structure(training_dataset.output_types,
                                   training_dataset.output_shapes)
next_element = iterator.get_next()

training_init_op = iterator.make_initializer(training_dataset)
validata_init_op = iterator.make_initializer(validation_dataset)

for _ in range(20):
    sess.run(training_init_op)
    for _ in range(100):
        sess.run(next_element)

    sess.run(validata_init_op)
    for _ in range(50):
        sess.run(next_element)

## 读取numpy数组
with np.load("/var/data/training_data.npy") as data:
    features = data["features"]
    labels = data["labels"]
assert features.shape[0] == labels.shape[0]
dataset = tf.data.Dataset.from_tensor_slices((features, labels))

# 使用占位符
with np.load("../data/training_data.npy") as data:
    features = data["features"]
    labels = data["labels"]

assert features.shape[0] == labels.shape[0]

features_placeholder = tf.placeholder(features.dtype, features.shape)
labels_placeholder = tf.placeholder(labels.dtype, labels.shape)

dataset = tf.data.Dataset.from_tensor_slices((features_placeholder, labels_placeholder))
iterator = dataset.make_initializable_iterator()

sess.run(iterator.initializer, feed_dict={features_placeholder: features, labels_placeholder: labels})

## 读取TFRecord数据
filenames = ['../data/file1.tfrecord', '../data/file2.tfrecord']
dataset = tf.data.TFRecordDataset(filenames)

# 使用占位符
filenames = tf.placeholder(tf.string, shape=[None])
dataset = tf.data.TFRecordDataset(filenames)
dataset = dataset.map(...)
dataset = dataset.repeat()
dataset = dataset.batch(32)
iterator = dataset.make_initializable_iterator()

# Initialize `iterator` with training data.
training_filenames = ["/var/data/file1.tfrecord", "/var/data/file2.tfrecord"]
sess.run(iterator.initializer, feed_dict={filenames: training_filenames})

# Initialize `iterator` with validation data.
validation_filenames = ["/var/data/validation1.tfrecord"]
sess.run(iterator.initializer, feed_dict={filenames: validation_filenames})

## 读取文本数据
filenames = ["../data/file1.txt", "../data/file2.txt"]
dataset = tf.data.Dataset.from_tensor_slices(filenames)
dataset = dataset.flat_map(
    lambda filename: (
        tf.data.TextLineDataset(filename)
            .skip(1)  # 跳过第一行
            .filter(lambda line: tf.not_equal(tf.substr(line, 0, 1), "#"))))  # 过滤掉前面有#号的


