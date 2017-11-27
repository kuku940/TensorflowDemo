#!/usr/bin/env python
# _*_ coding:utf-8 _*_

import tensorflow as tf

vocabulary_size = 5
embedding_size = 10
word_embeddings = tf.get_variable("word_embeddings", [vocabulary_size, embedding_size])
embedded_word_ids = tf.nn.embedding_lookup(word_embeddings, word_ids)
