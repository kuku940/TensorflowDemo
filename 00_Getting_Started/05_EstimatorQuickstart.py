#!/usr/bin/env python
# _*_ coding:utf-8 _*_

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
from six.moves.urllib.request import urlopen

import numpy as np
import tensorflow as tf

# Data sets
IRIS_TRAINING = '../data/iris/iris_training.csv'
IRIS_TRAINING_URL = 'http://download.tensorflow.org/data/iris_training.csv'

IRIS_TEST = "../data/iris/iris_test.csv"
IRIS_TEST_URL = "http://download.tensorflow.org/data/iris_test.csv"


def main():
    # 如果数据集不存在，那么下载并保存到本地
    if not os.path.exists(IRIS_TRAINING):
        raw = urlopen(IRIS_TRAINING_URL).read()
        with open(IRIS_TRAINING, 'wb') as f:
            f.write(raw)

    if not os.path.exists(IRIS_TEST):
        raw = urlopen(IRIS_TEST_URL).read()
        with open(IRIS_TEST, 'wb') as f:
            f.write(raw)

    # 读取数据
    training_set = tf.contrib.learn.datasets.base.load_csv_with_header(filename=IRIS_TRAINING, target_dtype=np.int,
                                                                       features_dtype=np.float32)
    test_set = tf.contrib.learn.datasets.base.load_csv_with_header(filename=IRIS_TEST, target_dtype=np.int,
                                                                   features_dtype=np.float32)

    # 创建特征向量
    feature_columns = [tf.feature_column.numeric_column("x", shape=[4])]
    # 创建3层DNN,分别是10, 20, 10
    classifier = tf.estimator.DNNClassifier(feature_columns=feature_columns, hidden_units=[10, 20, 10], n_classes=3,
                                            model_dir="../data/tmp/iris_model")
    # 定义训练集输入数据
    train_input_fn = tf.estimator.inputs.numpy_input_fn(x={"x": np.array(training_set.data)},
                                                        y=np.array(training_set.target), num_epochs=None, shuffle=True)
    # 训练模型
    classifier.train(input_fn=train_input_fn, steps=2000)

    # 定义测试集输入数据
    test_input_fn = tf.estimator.inputs.numpy_input_fn(x={"x": np.array(test_set.data)},
                                                       y=np.array(test_set.target), num_epochs=1, shuffle=False)
    # 计算准确度
    accuracy_score = classifier.evaluate(input_fn=test_input_fn)["accuracy"]
    print("\nTest Accuracy:{0:f}\n".format(accuracy_score))

    # 给两朵花的sample进行分类
    new_samples = np.array([[6.4, 3.2, 4.5, 1.5], [5.8, 3.1, 5.0, 1.7]], dtype=np.float32)
    predict_input_fn = tf.estimator.inputs.numpy_input_fn(x={"x": new_samples}, num_epochs=1, shuffle=False)

    predictions = list(classifier.predict(input_fn=predict_input_fn))
    predicted_classes = [p["classes"] for p in predictions]

    print("New Samples, Class Predictions:  {}\n"
          .format(predicted_classes))


if __name__ == "__main__":
    main()
