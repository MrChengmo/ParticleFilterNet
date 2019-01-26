import tensorflow as tf
import numpy as np

_L2_SCALE = 1.0  # scale when adding to loss instead of a global scaler


def weigth_variable(shape, layer_i, name=None):
    # stddev : 正态分布的标准差
    if name is None:
        name = "l%d_W" % (layer_i)
    initial = lambda: tf.truncated_normal(shape, stddev=0.1, name=name, dtype=tf.float64)  # 截断正态分布
    w = tf.Variable(initial_value=initial, dtype=tf.float64)
    return w


# 计算biases
def bias_varibale(shape, layer_i, name=None):
    if name is None:
        name = "l%d_B" % (layer_i)
    initial = lambda: tf.constant(0.1, shape=shape, name=name, dtype=tf.float64)
    b = tf.Variable(initial_value=initial,dtype=tf.float64)
    return b


# 计算卷积
def conv2d(x, W, layer_i, name=None):
    if name is None:
        name = "l%d_conv" % (layer_i)

    return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')


# 定义池化
def max_pool_2x2(x):
    return tf.nn.max_pool(x, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')


def max_pool_4x4(x):
    return tf.nn.max_pool(x, ksize=[1, 4, 4, 1], strides=[1, 4, 4, 1], padding='SAME')


def cnn(x_image, keep_prob):
    # layer1
    W_conv1 = weigth_variable([5, 5, 3, 32], layer_i=1)
    b_conv1 = bias_varibale([32], layer_i=1)
    h_conv1 = tf.nn.relu(conv2d(x_image, W_conv1, layer_i=1) + b_conv1)  # 100*100*32
    h_pool1 = max_pool_2x2(h_conv1)
    # layer2
    W_conv2 = weigth_variable([5, 5, 32, 64], layer_i=2)
    b_conv2 = bias_varibale([64], layer_i=2)
    h_conv2 = tf.nn.relu(conv2d(h_pool1, W_conv2, layer_i=2) + b_conv2)  # 100*100*32
    h_pool2 = max_pool_2x2(h_conv2)
    # layer3
    W_conv3 = weigth_variable([5, 5, 64, 16], layer_i=3)
    b_conv3 = bias_varibale([16], layer_i=3)
    h_conv3 = tf.nn.relu(conv2d(h_pool2, W_conv3, layer_i=3) + b_conv3)
    h_pool3 = max_pool_4x4(h_conv3)
    h_pool3 = tf.nn.dropout(h_pool3, keep_prob)
    # 四层全连接层
    W_fc1 = weigth_variable([7 * 7 * 16, 120], layer_i=4)
    b_fc1 = bias_varibale([120], layer_i=4)
    h_pool3_flat = tf.reshape(h_pool3, [-1, 7 * 7 * 16])
    h_fc1 = tf.nn.relu(tf.matmul(h_pool3_flat, W_fc1) + b_fc1)
    h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob)  # 防止过度拟合
    # 第五层全连接层
    W_fc2 = weigth_variable([120, 11], layer_i=5)
    b_fc2 = bias_varibale([11], layer_i=5)
    prediction = tf.nn.softmax(tf.matmul(h_fc1_drop, W_fc2) + b_fc2)
    return prediction


# Helper functions for constructing layers
def dense_layer(units, activation=None, use_bias=False, name=None):
    fn = lambda x: tf.layers.dense(
        x, units, activation=convert_activation_string(activation), use_bias=use_bias, name=name,
        kernel_regularizer=tf.contrib.layers.l2_regularizer(scale=_L2_SCALE))
    return fn


def conv2_layer(filters, kernel_size, activation=None, padding='same', strides=(1, 1), dilation_rate=(1, 1),
                data_format='channels_last', use_bias=False, name=None, layer_i=0, name2=None):
    if name is None:
        name = "l%d_conv%d" % (layer_i, np.max(kernel_size))
        if np.max(dilation_rate) > 1:
            name += "_d%d" % np.max(dilation_rate)
        if name2 is not None:
            name += "_" + name2
    fn = lambda x: tf.layers.conv2d(
        x, filters, kernel_size, activation=convert_activation_string(activation),
        padding=padding, strides=strides, dilation_rate=dilation_rate, data_format=data_format,
        kernel_regularizer=tf.contrib.layers.l2_regularizer(scale=_L2_SCALE),
        kernel_initializer=tf.variance_scaling_initializer(),
        use_bias=use_bias, name=name)
    return fn


def locallyconn2_layer(filters, kernel_size, activation=None, padding='same', strides=(1, 1), dilation_rate=(1, 1),
                       data_format='channels_last', use_bias=False, name=None, layer_i=0, name2=None):
    assert dilation_rate == (1, 1)  # keras layer doesnt have this input. maybe different name?
    if name is None:
        name = "l%d_conv%d" % (layer_i, np.max(kernel_size))
        if np.max(dilation_rate) > 1:
            name += "_d%d" % np.max(dilation_rate)
        if name2 is not None:
            name += "_" + name2
    fn = tf.keras.layers.LocallyConnected2D(
        filters, kernel_size, activation=convert_activation_string(activation),
        padding=padding, strides=strides, data_format=data_format,
        kernel_regularizer=tf.contrib.layers.l2_regularizer(scale=_L2_SCALE),
        kernel_initializer=tf.variance_scaling_initializer(),
        use_bias=use_bias, name=name)
    return fn


def convert_activation_string(activation):
    if isinstance(activation, str):
        if activation == 'relu':
            activation = tf.nn.relu
        elif activation == 'tanh':
            activation = tf.nn.tanh
        else:
            assert False
    return activation
