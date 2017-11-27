#!/usr/bin/env python
# _*_ coding:utf-8 _*_

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import itertools

import pandas as pd
import tensorflow as tf

tf.logging.set_verbosity(tf.logging.INFO)
# 犯罪率,住宅用户25000平方英尺以上,属于非零售业务的土地部分,一氧化碳浓度,住宅房间数,住宅年限,距离波士顿就业中心,税率,师生比率,住宅中值
COLUMNS = ["crim", "zn", "indus", "nox", "rm", "age", "dis", "tax", "ptratio", "medv"]
FEATURES = ["crim", "zn", "indus", "nox", "rm", "age", "dis", "tax", "ptratio"]
LABEL = "medv"

# 导入数据
training_set = pd.read_csv("../data/boston/boston_train.csv", skipinitialspace=True, skiprows=1, names=COLUMNS)
test_set = pd.read_csv("../data/boston/boston_test.csv", skipinitialspace=True, skiprows=1, names=COLUMNS)
prediction_set = pd.read_csv("../data/boston/boston_predict.csv", skipinitialspace=True, skiprows=1, names=COLUMNS)

# 定义回归模型
feature_cols = [tf.feature_column.numeric_column(k) for k in FEATURES]
regressor = tf.estimator.DNNRegressor(feature_columns=feature_cols, hidden_units=[10, 10],
                                      model_dir="../data/tmp/boston_model")


# 建立input_fn
def get_input_fn(data_set, num_epochs=None, shuffer=True):
    return tf.estimator.inputs.pandas_input_fn(
        x=pd.DataFrame({k: data_set[k].values for k in FEATURES}),
        y=pd.Series(data_set[LABEL].values),
        num_epochs=num_epochs,
        shuffle=shuffer
    )


regressor.train(input_fn=get_input_fn(training_set), steps=5000)

ev = regressor.evaluate(input_fn=get_input_fn(test_set, num_epochs=1, shuffer=False))

loss_score = ev['loss']

print("\nTest Loss:{0:f}\n".format(loss_score))

y = regressor.predict(input_fn=get_input_fn(prediction_set, num_epochs=1, shuffer=False))
predictions = list(p["predictions"] for p in itertools.islice(y, 6))
print("Predictions: {}".format(str(predictions)))
