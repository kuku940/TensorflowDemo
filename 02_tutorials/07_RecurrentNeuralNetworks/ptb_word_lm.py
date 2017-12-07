# Copyright 2015 The TensorFlow Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

"""Example / benchmark for building a PTB LSTM model.

Trains the model described in:
(Zaremba, et. al.) Recurrent Neural Network Regularization
http://arxiv.org/abs/1409.2329

There are 3 supported model configurations:
===========================================
| config | epochs | train | valid  | test
===========================================
| small  | 13     | 37.99 | 121.39 | 115.91
| medium | 39     | 48.45 |  86.16 |  82.07
| large  | 55     | 37.87 |  82.62 |  78.29
The exact results may vary depending on the random initialization.

The hyperparameters used in the model:
- init_scale - the initial scale of the weights
- learning_rate - the initial value of the learning rate
- max_grad_norm - the maximum permissible norm of the gradient
- num_layers - the number of LSTM layers
- num_steps - the number of unrolled steps of LSTM
- hidden_size - the number of LSTM units
- max_epoch - the number of epochs trained with the initial learning rate
- max_max_epoch - the total number of epochs for training
- keep_prob - the probability of keeping weights in the dropout layer
- lr_decay - the decay of the learning rate for each epoch after "max_epoch"
- batch_size - the batch size
- rnn_mode - the low level implementation of lstm cell: one of CUDNN,
             BASIC, or BLOCK, representing cudnn_lstm, basic_lstm, and
             lstm_block_cell classes.

The data required for this example is in the data/ dir of the
PTB dataset from Tomas Mikolov's webpage:

$ wget http://www.fit.vutbr.cz/~imikolov/rnnlm/simple-examples.tgz
$ tar xvf simple-examples.tgz

To run:

$ python ptb_word_lm.py --data_path=simple-examples/data/

"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import time

import numpy as np
import tensorflow as tf

import reader
import util

from tensorflow.python.client import device_lib

flags = tf.flags
logging = tf.logging

flags.DEFINE_string(
    "model", "small",
    "A type of model. Possible options are: small, medium, large.")
flags.DEFINE_string("data_path", None,
                    "Where the training/test data is stored.")
flags.DEFINE_string("save_path", None,
                    "Model output directory.")
flags.DEFINE_bool("use_fp16", False,
                  "Train using 16-bit floats instead of 32bit floats")
flags.DEFINE_integer("num_gpus", 1,
                     "If larger than 1, Grappler AutoParallel optimizer "
                     "will create multiple training replicas with each GPU "
                     "running one replica.")
flags.DEFINE_string("rnn_mode", None,
                    "The low level implementation of lstm cell: one of CUDNN, "
                    "BASIC, and BLOCK, representing cudnn_lstm, basic_lstm, "
                    "and lstm_block_cell classes.")
FLAGS = flags.FLAGS
BASIC = "basic"
CUDNN = "cudnn"
BLOCK = "block"


def data_type():
    return tf.float16 if FLAGS.use_fp16 else tf.float32


class PTBInput(object):
    """
    语言模型处理输入数据class
    设置参数config中的batch_size,num_steps到本地变量
    """

    def __init__(self, config, data, name=None):
        self.batch_size = batch_size = config.batch_size
        self.num_steps = num_steps = config.num_steps  # LSTM的展开步数
        self.epoch_size = ((len(data) // batch_size) - 1) // num_steps  # 每epoch内训练迭代轮数
        self.input_data, self.targets = reader.ptb_producer(
            data, batch_size, num_steps, name=name)  #


class PTBModel(object):
    """定义语言模型处理输入数据"""

    def __init__(self, is_training, config, input_):
        """
        初始化方法
        :param is_training: 训练标记
        :param config: 配置参数
        :param input_: PTBInput类实例
        """
        self._is_training = is_training
        self._input = input_
        self._rnn_params = None
        self._cell = None
        self.batch_size = input_.batch_size
        self.num_steps = input_.num_steps
        size = config.hidden_size  # config hidden_size(LSTM节点数)
        vocab_size = config.vocab_size  # 词汇表大小

        # 创建网络词嵌入embedding,将one-hot编码格式单词转向量表达形式
        with tf.device("/cpu:0"):  # 计算限定CPU进行
            # 初始化embedding矩阵，
            #   行数设词汇表数vocab_size，
            #   列数(单词向量表达维数)hidden_size,和LST单元包含节点数一致
            embedding = tf.get_variable(
                "embedding", [vocab_size, size], dtype=data_type())
            # tf.nn.embedding_lookup查询单对应向量表达获得inputs
            inputs = tf.nn.embedding_lookup(embedding, input_.input_data)

        # 训练状态加一层Dropout
        if is_training and config.keep_prob < 1:
            inputs = tf.nn.dropout(inputs, config.keep_prob)

        output, state = self._build_rnn_graph(inputs, config, is_training)

        # Softmax层，定义权重softmax_w、偏置softmax_b
        # tf.matmul 输出output乘权重加偏置得网络最后输出logits
        softmax_w = tf.get_variable(
            "softmax_w", [size, vocab_size], dtype=data_type())
        softmax_b = tf.get_variable("softmax_b", [vocab_size], dtype=data_type())
        logits = tf.nn.xw_plus_b(output, softmax_w, softmax_b)
        # Reshape logits to be a 3-D tensor for sequence loss
        logits = tf.reshape(logits, [self.batch_size, self.num_steps, vocab_size])

        # Use the contrib sequence loss and average over the batches
        loss = tf.contrib.seq2seq.sequence_loss(
            logits,
            input_.targets,
            tf.ones([self.batch_size, self.num_steps], dtype=data_type()),
            average_across_timesteps=False,
            average_across_batch=True)

        # tf.reduce_sum汇总batch误差，计算平均样本误差cost
        # 保留最终状态final_state
        self._cost = tf.reduce_sum(loss)
        self._final_state = state

        # 不是训练状态直接返回
        if not is_training:
            return

        # 学习速率lr,不可训练
        self._lr = tf.Variable(0.0, trainable=False)
        tvars = tf.trainable_variables()  # 获取全部可训练参数tvars
        # 计算tvars梯度,tf.clip_by_global_norm设梯度最大范数，起正则化效果
        grads, _ = tf.clip_by_global_norm(tf.gradients(self._cost, tvars),
                                          config.max_grad_norm)
        optimizer = tf.train.GradientDescentOptimizer(self._lr)
        self._train_op = optimizer.apply_gradients(
            zip(grads, tvars),
            global_step=tf.train.get_or_create_global_step())

        # _new_lr 控制学习速率
        self._new_lr = tf.placeholder(
            tf.float32, shape=[], name="new_learning_rate")
        # 定义_lr_update,tf
        self._lr_update = tf.assign(self._lr, self._new_lr)

    def _build_rnn_graph(self, inputs, config, is_training):
        if config.rnn_mode == CUDNN:
            return self._build_rnn_graph_cudnn(inputs, config, is_training)
        else:
            return self._build_rnn_graph_lstm(inputs, config, is_training)

    def _build_rnn_graph_cudnn(self, inputs, config, is_training):
        """Build the inference graph using CUDNN cell."""
        inputs = tf.transpose(inputs, [1, 0, 2])
        self._cell = tf.contrib.cudnn_rnn.CudnnLSTM(
            num_layers=config.num_layers,
            num_units=config.hidden_size,
            input_size=config.hidden_size,
            dropout=1 - config.keep_prob if is_training else 0)
        params_size_t = self._cell.params_size()
        self._rnn_params = tf.get_variable(
            "lstm_params",
            initializer=tf.random_uniform(
                [params_size_t], -config.init_scale, config.init_scale),
            validate_shape=False)
        c = tf.zeros([config.num_layers, self.batch_size, config.hidden_size],
                     tf.float32)
        h = tf.zeros([config.num_layers, self.batch_size, config.hidden_size],
                     tf.float32)
        self._initial_state = (tf.contrib.rnn.LSTMStateTuple(h=h, c=c),)
        outputs, h, c = self._cell(inputs, h, c, self._rnn_params, is_training)
        outputs = tf.transpose(outputs, [1, 0, 2])
        outputs = tf.reshape(outputs, [-1, config.hidden_size])
        return outputs, (tf.contrib.rnn.LSTMStateTuple(h=h, c=c),)

    def _get_lstm_cell(self, config, is_training):
        """
        获取LSTM单元
        :param config:
        :param is_training:
        :return:
        """
        if config.rnn_mode == BASIC:
            """
            设置tf.contrib.rnn.BasicLSTMCell为默认LSTM单元
            num_units: LSTM cell层中单元数 - config.hidder_size
            forget_bias: forget gates中的偏置
            state_is_tuple: True表示返回(c_state, m_state)的二元组
            activation: 状态之间转移的激活函数
            reuse: python布尔值，描述是否重用现有作用域的变量
            """
            return tf.contrib.rnn.BasicLSTMCell(
                config.hidden_size, forget_bias=0.0, state_is_tuple=True,
                reuse=not is_training)
        if config.rnn_mode == BLOCK:
            return tf.contrib.rnn.LSTMBlockCell(
                config.hidden_size, forget_bias=0.0)
        raise ValueError("rnn_mode %s not supported" % config.rnn_mode)

    def _build_rnn_graph_lstm(self, inputs, config, is_training):
        """Build the inference graph using canonical LSTM cells."""

        # Slightly better results can be obtained with forget gate biases
        # initialized to 1 but the hyperparameters of the model would need to be
        # different than reported in the paper.
        def make_cell():
            cell = self._get_lstm_cell(config, is_training)
            # 使用DropoutWrapper类来实现dropout功能，input_keep_pron控制输出的dropout概率
            if is_training and config.keep_prob < 1:
                cell = tf.contrib.rnn.DropoutWrapper(
                    cell, output_keep_prob=config.keep_prob)
            return cell

        # 使用 MultiRNNCell 类实现深层循环网络中每一个时刻的前向传播过程
        # number_of_layers 表示有多少层
        cell = tf.contrib.rnn.MultiRNNCell(
            [make_cell() for _ in range(config.num_layers)], state_is_tuple=True)

        # 初始化LSTM网络节点，参数为尺寸
        self._initial_state = cell.zero_state(config.batch_size, data_type())
        state = self._initial_state
        # Simplified version of tensorflow_models/tutorials/rnn/rnn.py's rnn().
        # This builds an unrolled LSTM for tutorial purposes only.
        # In general, use the rnn() or state_saving_rnn() from rnn.py.
        #
        # The alternative version of the code below is:
        #
        # inputs = tf.unstack(inputs, num=num_steps, axis=1)
        # outputs, state = tf.contrib.rnn.static_rnn(cell, inputs,
        #                            initial_state=self._initial_state)
        outputs = []
        with tf.variable_scope("RNN"):
            for time_step in range(self.num_steps):
                # 从第2次循环，tf.get_varible_scope.reuse_variables设置复用变量
                if time_step > 0: tf.get_variable_scope().reuse_variables()
                # 每次循环，传入inputs、state到堆叠LSTM单元(cell)
                # inputs 3维度，第1维 batch第几个样本，第2维 样本第几个单词，第3维 单词向量表达维度
                # inputs[:,time_step,:] 所有样本第time_step个单词
                (cell_output, state) = cell(inputs[:, time_step, :], state)

                # 结果cell_output添加输出列表outputs
                outputs.append(cell_output)

        # tf.concat串接output内容，tf.reshape转一维向量
        output = tf.reshape(tf.concat(outputs, 1), [-1, config.hidden_size])
        return output, state

    def assign_lr(self, session, lr_value):
        session.run(self._lr_update, feed_dict={self._new_lr: lr_value})

    def export_ops(self, name):
        """Exports ops to collections."""
        self._name = name
        ops = {util.with_prefix(self._name, "cost"): self._cost}
        if self._is_training:
            ops.update(lr=self._lr, new_lr=self._new_lr, lr_update=self._lr_update)
            if self._rnn_params:
                ops.update(rnn_params=self._rnn_params)
        for name, op in ops.items():
            tf.add_to_collection(name, op)
        self._initial_state_name = util.with_prefix(self._name, "initial")
        self._final_state_name = util.with_prefix(self._name, "final")
        util.export_state_tuples(self._initial_state, self._initial_state_name)
        util.export_state_tuples(self._final_state, self._final_state_name)

    def import_ops(self):
        """Imports ops from collections."""
        if self._is_training:
            self._train_op = tf.get_collection_ref("train_op")[0]
            self._lr = tf.get_collection_ref("lr")[0]
            self._new_lr = tf.get_collection_ref("new_lr")[0]
            self._lr_update = tf.get_collection_ref("lr_update")[0]
            rnn_params = tf.get_collection_ref("rnn_params")
            if self._cell and rnn_params:
                params_saveable = tf.contrib.cudnn_rnn.RNNParamsSaveable(
                    self._cell,
                    self._cell.params_to_canonical,
                    self._cell.canonical_to_params,
                    rnn_params,
                    base_variable_scope="Model/RNN")
                tf.add_to_collection(tf.GraphKeys.SAVEABLE_OBJECTS, params_saveable)
        self._cost = tf.get_collection_ref(util.with_prefix(self._name, "cost"))[0]
        num_replicas = FLAGS.num_gpus if self._name == "Train" else 1
        self._initial_state = util.import_state_tuples(
            self._initial_state, self._initial_state_name, num_replicas)
        self._final_state = util.import_state_tuples(
            self._final_state, self._final_state_name, num_replicas)

    """
    定义PTBModel class property
    @property装饰器 - 返回变量设只读，防止修改变量引发问题。
    input/initial_state/cost/final_state/lr/train_op
    """

    @property
    def input(self):
        return self._input

    @property
    def initial_state(self):
        return self._initial_state

    @property
    def cost(self):
        return self._cost

    @property
    def final_state(self):
        return self._final_state

    @property
    def lr(self):
        return self._lr

    @property
    def train_op(self):
        return self._train_op

    @property
    def initial_state_name(self):
        return self._initial_state_name

    @property
    def final_state_name(self):
        return self._final_state_name


class SmallConfig(object):
    """
    init_scale: 网络权重初始化scale
    learn_rate: 学习率初始值
    max_grad_norm: 梯度最大范数
    num_lyers: LSTM堆叠层数
    num_steps: LSTM梯度反向传播展开步数
    hidden_size: LSTM内隐含节点数
    max_epoch: 初始学习速率可训练epoch数，需要调整学习速率
    max_max_epoch: 总共可训练epoch数
    keep_prob: dropout层保留节点比例
    lr_decay: 学习率衰减速度
    batch_size: 每个batch样本数量
    """
    init_scale = 0.1
    learning_rate = 1.0
    max_grad_norm = 5
    num_layers = 2
    num_steps = 20
    hidden_size = 200
    max_epoch = 4
    max_max_epoch = 13
    keep_prob = 1.0
    lr_decay = 0.5
    batch_size = 20
    vocab_size = 10000
    rnn_mode = BLOCK


class MediumConfig(object):
    """Medium config - 中型模型
    减小init_scale,希望权重初值不要过大,小有利温和训练;
    学习率,最大梯度范数不变;
    LSTM层数不变;
    梯度反向传播展开步数num_steps从20增大到35;
    hidden_size、max_max_epoch增大3倍;
    设置dropout keep_prob=0.5;
    学习迭代次数增大;
    学习速率衰减速率lr_decay减小;
    batch_size、词汇表vocab_size不变。
    """
    init_scale = 0.05
    learning_rate = 1.0
    max_grad_norm = 5
    num_layers = 2
    num_steps = 35
    hidden_size = 650
    max_epoch = 6
    max_max_epoch = 39
    keep_prob = 0.5
    lr_decay = 0.8
    batch_size = 20
    vocab_size = 10000
    rnn_mode = BLOCK


class LargeConfig(object):
    """Large config. - 大型模型
    进一步缩小init_scale;
    放宽最大梯度范数max_grad_norm到10;
    hidden_size提升到1500;
    max_epoch、max_max_epoch增大;
    keep_prob因模型复杂度上升继续下降;
    学习速率衰减速率lr_decay进一步减小。
    """
    init_scale = 0.04
    learning_rate = 1.0
    max_grad_norm = 10
    num_layers = 2
    num_steps = 35
    hidden_size = 1500
    max_epoch = 14
    max_max_epoch = 55
    keep_prob = 0.35
    lr_decay = 1 / 1.15
    batch_size = 20
    vocab_size = 10000
    rnn_mode = BLOCK


class TestConfig(object):
    """Tiny config, for testing.
    参数尽量最小值
    """
    init_scale = 0.1
    learning_rate = 1.0
    max_grad_norm = 1
    num_layers = 1
    num_steps = 2
    hidden_size = 2
    max_epoch = 1
    max_max_epoch = 1
    keep_prob = 1.0
    lr_decay = 0.5
    batch_size = 20
    vocab_size = 10000
    rnn_mode = BLOCK


def run_epoch(session, model, eval_op=None, verbose=False):
    """Runs the model on the given data."""
    start_time = time.time()
    costs = 0.0
    iters = 0
    state = session.run(model.initial_state)

    fetches = {
        "cost": model.cost,
        "final_state": model.final_state,
    }
    if eval_op is not None:
        fetches["eval_op"] = eval_op

    for step in range(model.input.epoch_size):
        feed_dict = {}
        for i, (c, h) in enumerate(model.initial_state):
            feed_dict[c] = state[i].c
            feed_dict[h] = state[i].h

        vals = session.run(fetches, feed_dict)
        cost = vals["cost"]
        state = vals["final_state"]

        costs += cost
        iters += model.input.num_steps

        if verbose and step % (model.input.epoch_size // 10) == 10:
            print("%.3f perplexity: %.3f speed: %.0f wps" %
                  (step * 1.0 / model.input.epoch_size, np.exp(costs / iters),
                   iters * model.input.batch_size * max(1, FLAGS.num_gpus) /
                   (time.time() - start_time)))

    return np.exp(costs / iters)


def get_config():
    """Get model config."""
    config = None
    if FLAGS.model == "small":
        config = SmallConfig()
    elif FLAGS.model == "medium":
        config = MediumConfig()
    elif FLAGS.model == "large":
        config = LargeConfig()
    elif FLAGS.model == "test":
        config = TestConfig()
    else:
        raise ValueError("Invalid model: %s", FLAGS.model)
    if FLAGS.rnn_mode:
        config.rnn_mode = FLAGS.rnn_mode
    if FLAGS.num_gpus != 1 or tf.__version__ < "1.3.0":
        config.rnn_mode = BASIC
    return config


def main(_):
    if not FLAGS.data_path:
        raise ValueError("Must set --data_path to PTB data directory")
    gpus = [
        x.name for x in device_lib.list_local_devices() if x.device_type == "GPU"
    ]
    if FLAGS.num_gpus > len(gpus):
        raise ValueError(
            "Your machine has only %d gpus "
            "which is less than the requested --num_gpus=%d."
            % (len(gpus), FLAGS.num_gpus))

    # 将路径中的文件单词转换成为数值索引形式
    raw_data = reader.ptb_raw_data(FLAGS.data_path)
    train_data, valid_data, test_data, _ = raw_data

    #
    config = get_config()
    eval_config = get_config()
    eval_config.batch_size = 1
    eval_config.num_steps = 1

    with tf.Graph().as_default():
        initializer = tf.random_uniform_initializer(-config.init_scale,
                                                    config.init_scale)

        with tf.name_scope("Train"):
            train_input = PTBInput(config=config, data=train_data, name="TrainInput")
            with tf.variable_scope("Model", reuse=None, initializer=initializer):
                m = PTBModel(is_training=True, config=config, input_=train_input)
            tf.summary.scalar("Training Loss", m.cost)
            tf.summary.scalar("Learning Rate", m.lr)

        with tf.name_scope("Valid"):
            valid_input = PTBInput(config=config, data=valid_data, name="ValidInput")
            with tf.variable_scope("Model", reuse=True, initializer=initializer):
                mvalid = PTBModel(is_training=False, config=config, input_=valid_input)
            tf.summary.scalar("Validation Loss", mvalid.cost)

        with tf.name_scope("Test"):
            test_input = PTBInput(
                config=eval_config, data=test_data, name="TestInput")
            with tf.variable_scope("Model", reuse=True, initializer=initializer):
                mtest = PTBModel(is_training=False, config=eval_config,
                                 input_=test_input)

        models = {"Train": m, "Valid": mvalid, "Test": mtest}
        for name, model in models.items():
            model.export_ops(name)
        metagraph = tf.train.export_meta_graph()
        if tf.__version__ < "1.1.0" and FLAGS.num_gpus > 1:
            raise ValueError("num_gpus > 1 is not supported for TensorFlow versions "
                             "below 1.1.0")
        soft_placement = False
        if FLAGS.num_gpus > 1:
            soft_placement = True
            util.auto_parallel(metagraph, m)

    with tf.Graph().as_default():
        tf.train.import_meta_graph(metagraph)
        for model in models.values():
            model.import_ops()
        sv = tf.train.Supervisor(logdir=FLAGS.save_path)
        config_proto = tf.ConfigProto(allow_soft_placement=soft_placement)
        with sv.managed_session(config=config_proto) as session:
            for i in range(config.max_max_epoch):
                lr_decay = config.lr_decay ** max(i + 1 - config.max_epoch, 0.0)
                m.assign_lr(session, config.learning_rate * lr_decay)

                print("Epoch: %d Learning rate: %.3f" % (i + 1, session.run(m.lr)))
                train_perplexity = run_epoch(session, m, eval_op=m.train_op,
                                             verbose=True)
                print("Epoch: %d Train Perplexity: %.3f" % (i + 1, train_perplexity))
                valid_perplexity = run_epoch(session, mvalid)
                print("Epoch: %d Valid Perplexity: %.3f" % (i + 1, valid_perplexity))

            test_perplexity = run_epoch(session, mtest)
            print("Test Perplexity: %.3f" % test_perplexity)

            if FLAGS.save_path:
                print("Saving model to %s." % FLAGS.save_path)
                sv.saver.save(session, FLAGS.save_path, global_step=sv.global_step)


if __name__ == "__main__":
    tf.app.run()
