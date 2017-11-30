#!/usr/bin/env python
# _*_ coding:utf-8 _*_

"""
Basic Word2Vec Example
将单词转换为密集向量(Dense Vector)
Word2Vec,Word Embedding,词向量或词嵌入。语言字词转向量形式表达(Vector Representations)模型。
Word2Vec，计算非常高效，从原始语料学习字词空间向量预测模型，意思相近词向量空间位置接近。
    CBOW(Continuous Bag of Words)模式从原始语句推测目标字词，适合小型数据。
    Skip-Gram从目标字词推测原始语句，适合大型语料。
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import collections
import math
import os
import random
import zipfile
from tempfile import gettempdir

import numpy as np
import tensorflow as tf
from six.moves import urllib
from six.moves import xrange


# step1 - 下载数据
# pylint: disable=redefined-outer-name
def maybe_download(filename, expected_bytes):
    """
    下载数据集，并核对文件尺寸
    :param filename: 保存的文件名
    :param expected_bytes: 文件的字节数 - 确认文件没有损坏
    :return: 本地文件路径
    """
    url = "http://mattmahoney.net/dc/"

    # local_filename = os.path.join(gettempdir(), filename)
    local_filename = os.path.join("../../data/word2vec", filename)
    if not os.path.exists(local_filename):
        local_filename, _ = urllib.request.urlretrieve(url + filename, local_filename)

    statinfo = os.stat(local_filename)  # 返回文件的系统状态
    if statinfo.st_size == expected_bytes:  # 验证字节数是否正确，确保文件没有损坏
        print('Fount and verified', filename)
    else:
        print(statinfo.st_size)
        raise Exception('Failed to verify ' + local_filename + '. Can you get to it with a browser?')
    return local_filename


filename = maybe_download('text8.zip', 31344016)
print(filename)


def read_data(filename):
    """
    解压压缩文件，并转换成单词列表
    :param filename: 压缩文件路径
    :return: 分隔后的单词list
    """
    with zipfile.ZipFile(filename) as f:  # 读取压缩包中的所有文件信息
        # f.namelist - 返回压缩包内所有文件的信息
        # tf.compat.as_str - 将输入转换成为字符串
        data = tf.compat.as_str(f.read(f.namelist()[0])).split()
    return data


vocabulary = read_data(filename)
print("Data size: ", len(vocabulary))  # 17005207 words

# step2 - 创建dict,并用UNK替换罕见词
vocabulary_size = 50000


def build_dataset(words, n_words):
    """
    创建单词的数据集
    :param words: 单词集合
    :param n_words: 统计的词汇量
    :return:
        data: 对words中word替换成数值索引后的集合
        count: 单词+词频 集合
        dictionary: 单词+数值索引的字典
        reversed_dictionary: 数值索引+单词的字典
    """
    count = [['UNK', 1]]
    # 统计词条词频，并统计词频前50000的单词
    count.extend(collections.Counter(words).most_common(n_words - 1))
    dictionary = dict()
    # 遍历单词 并给单词一个数值索引，即：字典的角标
    for word, _ in count:
        dictionary[word] = len(dictionary)

    data = list()
    unk_count = 0
    # 统计低频词数量，并对其进行类似脱敏处理，即单词替换成数值索引
    for word in words:
        index = dictionary.get(word, 0)
        if index == 0:
            unk_count += 1
        data.append(index)
    count[0][1] = unk_count  # 设置UNK词频
    reversed_dictionary = dict(zip(dictionary.values(), dictionary.keys()))  # 翻转字典结构
    return data, count, dictionary, reversed_dictionary


# Filling 4 global variables:
# data - list of codes (integers from 0 to vocabulary_size-1).
#   This is the original text but words are replaced by their codes
# count - map of words(strings) to count of occurrences -
# dictionary - map of words(strings) to their codes(integers) - 单词数，[word - code]
# reverse_dictionary - maps codes(integers) to words(strings) - 单词数，[code - word]
data, count, dictionary, reverse_dictionary = build_dataset(vocabulary, vocabulary_size)

del vocabulary  # 删除原始单词列表，节约内存。
print("Most common words (+UNK)", count[:5])
print("Sample data", data[:10], [reverse_dictionary[i] for i in data[:10]])

data_index = 0


# Step3: 为skip-gram模型生成训练数据
def generate_batch(batch_size, num_skips, skip_window):
    """
    为skip-gram生成训练数据及标签[Skip-gram 目标词 -> 上下文]
    :param batch_size: Batch大小
    :param num_skips: 单词生成样本个数，不能大于skip_window两倍
    :param skip_window: 单词最远可联系距离，设1只能跟紧邻两个单词生成样本
    :return:
        batch: 目标词
        labels: 上下文
    """
    global data_index  # 单词序号
    assert batch_size % num_skips == 0
    assert num_skips <= 2 * skip_window

    # 初始化batch/labels数组
    batch = np.ndarray(shape=(batch_size), dtype=np.int32)
    labels = np.ndarray(shape=(batch_size, 1), dtype=np.int32)
    # 定义span单词创建样本单词数量，包括目标单词和前后单词[skip_window target skip_window]
    span = 2 * skip_window + 1
    buffer = collections.deque(maxlen=span)  # 双向队列
    if data_index + span > len(data):
        data_index = 0

    buffer.extend(data[data_index:data_index + span])
    data_index += span
    for i in range(batch_size // num_skips):
        # 拿到目标单词的上下文
        context_words = [w for w in range(span) if w != skip_window]
        words_to_use = random.sample(context_words, num_skips)
        for j, context_word in enumerate(words_to_use):
            # 将目标词和上下文组合
            batch[i * num_skips + j] = buffer[skip_window]
            labels[i * num_skips + j, 0] = buffer[context_word]
        if data_index == len(data):
            buffer[:] = data[:span]
            data_index = span
        else:
            buffer.append(data[data_index])
            data_index += 1

    # Backtrack a little bit to avoid skipping words in the end of a batch
    data_index = (data_index + len(data) - span) % len(data)
    return batch, labels


batch, labels = generate_batch(batch_size=8, num_skips=2, skip_window=1)
for i in range(8):
    print(batch[i], reverse_dictionary[batch[i]], '->', labels[i, 0], reverse_dictionary[labels[i, 0]])

# Step 4:Build and train a skip-grap model.
batch_size = 128
embedding_size = 128  # Dimension of the embedding vector - 词向量维度
skip_window = 1  # How many words to consider left and right - 上下文的大小
num_skips = 2  # How many time to reuse an input to generate a label - 目标词提取的样本数
num_sampled = 64  # Number of negative examples to sample 负样本噪音单词数

# We pick a random validation set to sample nearest neighbors. Here we limit the
# validation samples to the words that have a low numeric ID, which by
# construction are also the most frequent. These 3 variables are used only for
# displaying model accuracy, they don't affect calculation.
valid_size = 16  # 抽取16个单词作为验证集
valid_window = 100  # 指定词频最高的前100个单词
# 从词频最高的前100词中随机抽取16个; replace=False -> 不重复
valid_examples = np.random.choice(valid_window, valid_size, replace=False)

graph = tf.Graph()

with graph.as_default():
    # Input data
    train_inputs = tf.placeholder(tf.int32, shape=[batch_size])
    train_labels = tf.placeholder(tf.int32, shape=[batch_size, 1])
    valid_dataset = tf.constant(valid_examples, dtype=tf.int32)

    # Ops and
    with tf.device('/cpu:0'):
        # 随机生成所有单词词向量embeddings，单词表大小50000,向量维度128
        embeddings = tf.Variable(tf.random_uniform([vocabulary_size, embedding_size], -1.0, 1.0))
        # 从embeddings中获取训练集指定索引的词向量
        embed = tf.nn.embedding_lookup(embeddings, train_inputs)

        # 构建权重和偏置
        nce_weights = tf.Variable(tf.truncated_normal([vocabulary_size, embedding_size],
                                                      stddev=1.0 / math.sqrt(embedding_size)))
        nce_biases = tf.Variable(tf.zeros([vocabulary_size]))

    # Compute the average NCE loss for the batch.
    # tf.nce_loss automatically draws a new sample of the negative labels each
    # time we evaluate the loss.
    # Explanation of the meaning of NCE loss:
    #   http://mccormickml.com/2016/04/19/word2vec-tutorial-the-skip-gram-model/
    loss = tf.reduce_mean(tf.nn.nce_loss(weights=nce_weights, biases=nce_biases, labels=train_labels, inputs=embed,
                                         num_sampled=num_sampled, num_classes=vocabulary_size))
    # 创建学习率为1的SGD优化器
    optimizer = tf.train.GradientDescentOptimizer(1.0).minimize(loss)

    # Compute the cosine similarity between minibatch examples and all embeddings.
    # 查询验证单词嵌入向量，计算验证单词嵌入同与词汇表所有单词相似性。
    norm = tf.sqrt(tf.reduce_sum(tf.square(embeddings), 1, keep_dims=True))
    normalized_embeddings = embeddings / norm
    valid_embeddings = tf.nn.embedding_lookup(normalized_embeddings, valid_dataset)
    similarity = tf.matmul(valid_embeddings, normalized_embeddings, transpose_b=True)

    # Add variable initializer
    init = tf.global_variables_initializer()

# Step 5: Begin training.
num_steps = 100001

with tf.Session(graph=graph) as session:
    # We must initialize all variables before we use them
    init.run()
    print("Initialized")

    average_loss = 0
    for step in xrange(num_steps):
        batch_inputs, batch_labels = generate_batch(batch_size, num_skips, skip_window)

        # We preform one update step by evaluating the optimizer op
        # (including it in the list of returned values for session.run())
        _, loss_val = session.run([optimizer, loss], feed_dict={train_inputs: batch_inputs, train_labels: batch_labels})
        average_loss += loss_val

        if step % 2000 == 0:
            if step > 0:
                average_loss /= 2000
            # The average loss is an estimate of the loss over the last 2000 batches
            print("Average loss at step", step, ": ", average_loss)
            average_loss = 0

        # 计算验证单词和全部单词相似度，验证单词最相似8个单词展示。
        # Note that this is expensive (~20% slowdown if computed every 500 steps)
        if step % 10000 == 0:
            sim = similarity.eval()
            for i in xrange(valid_size):
                valid_word = reverse_dictionary[valid_examples[i]]
                top_k = 8  # number of nearest neighbors
                nearest = (-sim[i, :]).argsort()[1:top_k + 1]
                log_str = 'Nearest to %s:' % valid_word
                for k in xrange(top_k):
                    close_word = reverse_dictionary[nearest[k]]
                    log_str = '%s %s,' % (log_str, close_word)
                print(log_str)

        final_embeddings = normalized_embeddings.eval()


# Step 6: Visualize the embeddings
# pylint: disable = missing-docstring
def plot_with_labels(low_dim_embs, labels, filename):
    """
    function to draw visualization of distance between embedding
    :param low_dim_embs:
    :param labels:
    :param filename:
    :return:
    """
    assert low_dim_embs.shape[0] >= len(labels), 'More labels than embeddings'
    plt.figure(figsize=(18, 18))  # in inches
    for i, label in enumerate(labels):
        x, y = low_dim_embs[i, :]
        plt.scatter(x, y)
        plt.annotate(label, xy=(x, y), xytext=(5, 2), textcoords='offset points', ha='right', va='bottom')
    plt.savefig(filename)


try:
    # pylint: disable = g-import-not-at-top
    from sklearn.manifold import TSNE
    import matplotlib.pyplot as plt

    # sklearn.manifold.TSNE实现降维，原始128维词向量降到2维
    tsne = TSNE(perplexity=30, n_components=2, init='pca', n_iter=5000, method='exact')
    plot_only = 500
    low_dim_embs = tsne.fit_transform(final_embeddings[:plot_only, :])
    labels = [reverse_dictionary[i] for i in xrange(plot_only)]
    # plot_with_labels(low_dim_embs, labels, os.path.join(gettempdir(), 'tsne.png'))
    plot_with_labels(low_dim_embs, labels, os.path.join("../../data/tmp/word2vec", 'tsne.png'))

except ImportError as ex:
    print("Please install sklearn, matplotlib, and scipy to show embeddings.")
    print(ex)
