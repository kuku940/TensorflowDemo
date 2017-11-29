#!/usr/bin/env python
# _*_ coding:utf-8 _*_

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os

from six.moves import xrange  # pylint: disable=redefined-builtin
import tensorflow as tf

# 处理图片的尺寸，cifar10的大小为32x32
IMAGE_SIZE = 24

# CIFAR-10数据集
NUM_CLASSES = 10
NUM_EXAMPLES_PER_EPOCH_FOR_TRAIN = 50000
NUM_EXAMPLES_PER_EPOCH_FOR_EVAL = 10000


def read_cifar10(filename_queue):
    """
    从文件序列中读取数据，并处理
    :param filename_queue: 文件名队列
    :return: 带有标签的记录
    """

    class CIFAR10Record(object):
        pass

    result = CIFAR10Record()

    # CIFAR-10数据集图片的维度
    label_bytes = 1
    result.height = 32
    result.width = 32
    result.depth = 3
    image_bytes = result.height * result.width * result.depth
    # 每条记录占用的字节数
    record_bytes = image_bytes + label_bytes

    # 从文件中读取数据，CIFAR-10没有header和footer,设置header/footer为0
    reader = tf.FixedLengthRecordReader(record_bytes=record_bytes)
    result.key, value = reader.read(filename_queue)

    # 转换字符串为向量
    record_bytes = tf.decode_raw(value, tf.uint8)

    # 第一个字节是label -> int
    result.label = tf.cast(tf.strided_slice(record_bytes, [0], [label_bytes]), tf.int32)
    # 随后的字节是images -> [3,32,32]
    depth_major = tf.reshape(tf.strided_slice(record_bytes, [label_bytes], [label_bytes + image_bytes]),
                             [result.depth, result.height, result.width])
    # 转换[depth, height, width] -> [height, width, depth]
    result.uint8image = tf.transpose(depth_major, [1, 2, 0])
    return result


def _generate_image_and_label_batch(image, label, min_queue_examples, batch_size, shuffle):
    """
    构造批量处理的队列
    :param image: [height, width, 3] float32
    :param label: int32
    :param min_queue_examples:
    :param batch_size: batch的图片数量
    :param shuffle: 是否打乱
    :return:
        images: [batch_size, height, width, 3]
        labels: [batch_size]
    """
    num_preprocess_threads = 16
    if shuffle:
        images, label_batch = tf.train.shuffle_batch([image, label], batch_size=batch_size,
                                                     num_threads=num_preprocess_threads,
                                                     capacity=min_queue_examples + 3 * batch_size,
                                                     min_after_dequeue=min_queue_examples)
    else:
        images, label_batch = tf.train.batch([image, label], batch_size=batch_size, num_threads=num_preprocess_threads,
                                             capacity=min_queue_examples + 3 * batch_size)

    # 在可视化工具中显示训练集图片
    tf.summary.image("images", images)
    return images, tf.reshape(label_batch, [batch_size])


def distorted_inputs(data_dir, batch_size):
    """
    变换图片
    :param data_dir:
    :param batch_size:
    :return:
    """
    filenames = [os.path.join(data_dir, 'data_batch_%d.bin' % i) for i in range(1, 6)]
    for f in filenames:
        if not tf.gfile.Exists(f):
            raise ValueError('Faild to find file: ' + f)

    # 创建一个读取文件的队列
    filename_queue = tf.train.string_input_producer(filenames)

    # 读取文件队列中文件的记录
    read_input = read_cifar10(filename_queue)
    reshaped_image = tf.cast(read_input.uint8image, tf.float32)

    height = IMAGE_SIZE
    width = IMAGE_SIZE

    # 为训练模型处理图片 - 随机扭曲图片
    distorted_image = tf.random_crop(reshaped_image, [height, width, 3])  # 32 -> 24 裁剪图片
    distorted_image = tf.image.random_flip_left_right(distorted_image)  # 水平翻转图片

    # 改变亮度和对比度
    distorted_image = tf.image.random_brightness(distorted_image, max_delta=63)
    distorted_image = tf.image.random_contrast(distorted_image, lower=0.2, upper=1.8)

    # 标准化 减去平均值并除以方差
    float_image = tf.image.per_image_standardization(distorted_image)

    #
    float_image.set_shape([height, width, 3])
    read_input.label.set_shape([1])

    min_fraction_of_examples_in_queue = 0.4
    min_queue_examples = int(NUM_EXAMPLES_PER_EPOCH_FOR_TRAIN * min_fraction_of_examples_in_queue)
    print('Filling queue with %d CIFAR images before starting to train. '
          'This will take a few minutes.' % min_queue_examples)

    # Generate a batch of images and labels by building up a queue of examples.
    return _generate_image_and_label_batch(float_image, read_input.label,
                                           min_queue_examples, batch_size,
                                           shuffle=True)


def inputs(eval_data, data_dir, batch_size):
    """
    构造CIFAR评估数据
    :param eval_data: 
    :param data_dir: 数据集文件夹
    :param batch_size:
    :return:
    """
    if not eval_data:
        filenames = [os.path.join(data_dir, 'data_batch_%d.bin') % i for i in range(1, 6)]
        num_examples_per_epoch = NUM_EXAMPLES_PER_EPOCH_FOR_TRAIN
    else:
        filenames = [os.path.join(data_dir, 'test_batch.bin')]
        num_examples_per_epoch = NUM_EXAMPLES_PER_EPOCH_FOR_EVAL

    for f in filenames:
        if not tf.gfile.Exists(f):
            raise ValueError('Failed to find file: ' + f)

    # 构建队列
    filename_queue = tf.train.string_input_producer(filenames)

    # 从文件队列中去读记录
    read_input = read_cifar10(filename_queue)
    reshaped_image = tf.cast(read_input.uint8image, tf.float32)

    height = IMAGE_SIZE
    width = IMAGE_SIZE

    resized_image = tf.image.resize_image_with_crop_or_pad(reshaped_image, height, width)
    float_image = tf.image.per_image_standardization(resized_image)

    #
    float_image.set_shape([height, width, 3])
    read_input.label.set_shape([1])

    min_fraction_of_examples_in_queue = 0.4
    min_queue_examples = int(num_examples_per_epoch * min_fraction_of_examples_in_queue)

    return _generate_image_and_label_batch(float_image, read_input.label, min_queue_examples, batch_size, shuffle=False)
