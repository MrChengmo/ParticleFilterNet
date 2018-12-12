#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
PFcell

Created on Tue Nov 20 12:50:12 2018

@author: silence
"""
import numpy as np
import tensorflow as tf
from tensorflow.contrib.rnn import RNNCell as rnn
from tensorflow.contrib import image
from tensorflow.python.ops import math_ops
from PFlayer import conv2_layer, locallyconn2_layer, dense_layer


class PFCellClass(rnn):
    """
    PF-NET核心模块，使用RNN结构实现粒子滤波过程，RNN的隐状态对应于粒子集
    转移，观测及重采样模型被设计为可微的神经网络单元
    将粒子滤波算法视为计算图或可微分的程序
    Cell inputs: map,ins
    Cell states: particle_states,particle_weights
    Cell outputs: particle_states,particle_weights
    """

    def __init__(self, map_data, paramters, batch_size, particle_nums):
        super(PFCellClass, self).__init__()
        self._map = map_data
        self._parameters = paramters
        self._batch_size = batch_size
        self._particle_nums = particle_nums
        self._states_shape = (batch_size, particle_nums, 2)
        self._weights_shape = (batch_size, particle_nums)

    @property
    def state_size(self):
        return (tf.TensorShape(self._states_shape[1:]), tf.TensorShape(self._weights_shape[1:]))

    @property
    def output_size(self):
        return (tf.TensorShape(self._states_shape[1:]), tf.TensorShape(self._weights_shape[1:]))

    def __call__(self, inputs, states):
        with tf.variable_scope(tf.get_variable_scope()):
            particle_states, particle_weights = states

            # update particle by insX/insY
            new_particle_states = self.transitionModel(particle_states, inputs)

            # calc new weights by CNN
            new_particle_weights = self.obervationModel(self._map, particle_states, new_particle_states)
            particle_weights += new_particle_weights

            # resample particles
            new_states, new_weights = self.resampleModel(particle_states, particle_weights,
                                                         resample_para=self._parameters.resample_para)

            output = new_states, new_weights
            state = new_states, new_weights
        return output, state

    def transitionModel(self, particle_states, ins):
        """
        particle_states update by ins information and transition noise
        """
        distance_para_x = self._parameters.transition_para[0] * self._parameters.meter_pixel_para
        distance_para_y = self._parameters.transition_para[1] * self._parameters.meter_pixel_para
        with tf.name_scope('transition'):
            loc_x, loc_y = tf.unstack(particle_states, axis=-1, num=2)
            ins_x, ins_y = tf.unstack(ins, axis=-1, num=2)
            ins_x = tf.tile([ins_x], multiples=[self._particle_nums, 1])
            ins_x = tf.transpose(ins_x)
            ins_y = tf.tile([ins_y], multiples=[self._particle_nums, 1])
            ins_y = tf.transpose(ins_y)
            ins_x += tf.to_double(
                tf.random_normal(loc_x.get_shape(), mean=0.0, stddev=self._parameters.step_stddev) * distance_para_x)
            ins_y += tf.to_double(
                tf.random_normal(loc_y.get_shape(), mean=0.0, stddev=self._parameters.step_stddev) * distance_para_y)

            new_particle_states = tf.stack([loc_x + ins_x, loc_y + ins_y], axis=-1)
        return new_particle_states

    def obervationModel(self, map_data, old_particle_states, new_particle_states):
        """
        this model transform the big map to particle maps for each particle,where a particle map 
        is a local view from the state defined by the particle.And we use Cnn to calc the particle
        weight by particle maps
        """

        particle_maps = self.mapTransform(self._map, old_particle_states, new_particle_states)
        particle_maps = tf.reshape(particle_maps, [self._batch_size * self._particle_nums]
                                   + particle_maps.shape.as_list()[2:])
        map_features = self.mapFeatures(particle_maps)

        weight_vec = tf.reshape(map_features, [self._batch_size * self._particle_nums, -1])
        weight_vec = self.vectorFeatures(weight_vec)

        new_particle_weights = tf.reshape(weight_vec, [self._batch_size, self._particle_nums])
        return new_particle_weights

    @staticmethod
    def resampleModel(particle_states, particle_weights, resample_para):
        with tf.name_scope('resample'):
            assert 0.0 < resample_para <= 1.0
            batch_size, num_particles = particle_states.get_shape().as_list()[:2]

            # normalize
            particle_weights = particle_weights - tf.reduce_logsumexp(particle_weights,
                                                                      axis=-1, keep_dims=True)

            uniform_weights = tf.constant(-np.log(num_particles),
                                          shape=(batch_size, num_particles), dtype=tf.float64)

            # build sampling distribution, q(s), and update particle weights
            if resample_para < 1.0:
                # soft resampling
                q_weights = tf.stack([particle_weights + np.log(resample_para),
                                      uniform_weights + np.log(1.0 - resample_para)], axis=-1)
                q_weights = tf.reduce_logsumexp(q_weights, axis=-1, keep_dims=False)
                q_weights = q_weights - tf.reduce_logsumexp(q_weights, axis=-1, keep_dims=True)  # normalized

                particle_weights = particle_weights - q_weights  # this is unnormalized
            else:
                # hard resampling. this will produce zero gradients
                q_weights = particle_weights
                particle_weights = uniform_weights

            # sample particle indices according to q(s)
            indices = tf.cast(tf.multinomial(q_weights, num_particles), tf.int32)  # shape: (batch_size, num_particles)

            # index into particles
            helper = tf.range(0, batch_size * num_particles, delta=num_particles, dtype=tf.int32)  # (batch, )
            indices = indices + tf.expand_dims(helper, axis=1)

            particle_states = tf.reshape(particle_states, (batch_size * num_particles, 2))
            particle_states = tf.gather(particle_states, indices=indices, axis=0)  # (batch_size, num_particles, 2)

            particle_weights = tf.reshape(particle_weights, (batch_size * num_particles,))
            particle_weights = tf.gather(particle_weights, indices=indices, axis=0)  # (batch_size, num_particles,)

        return particle_states, particle_weights

    def mapTransform(self, map_data, old_particle_states, new_particle_states):
        """
        :map_data: tf op Big IndoorMap
        :param old_particle_states: tf op (batch, K, 2), time t-1 's particle states
        :param new_particle_states: tf op (batch, K, 2), time   t 's transition particle states
        """
        batch_size, num_particles = old_particle_states.get_shape().as_list()[:2]
        total_samples = batch_size * num_particles
        old_flat_states = tf.reshape(old_particle_states, [total_samples, 2])
        new_flat_states = tf.reshape(new_particle_states, [total_samples, 2])
        particle_map_list = []
        for i in range(total_samples):
            particle_map_list.append(self.mapCut(map_data, self._parameters.particle_map_length,
                                                 old_flat_states[i], new_flat_states[i]))
        particle_map = tf.reshape(particle_map_list, [batch_size, num_particles,
                                                      self._parameters.particle_map_shape[0],
                                                      self._parameters.particle_map_shape[1], 3])

        return particle_map

    def mapCut(self, map_data, particle_map_length, old_partcile_states, new_particle_states):
        """
        给定前后坐标，根据运动方向旋转并裁剪地图，得到基于当前坐标的局部地图
        :param map_data: 原始地图数据 Tensor[height,width,channel=3] int
        :param particle_map_length: 局部地图的形状大小，默认正方形 Tensor[1] float32
        :param old_partcile_states: 前一时刻的坐标 Tensor[2](loc_x,loc_y) float32
        :param new_particle_states: 当前时刻的坐标 Tensor[2](loc_x,loc_y) float32
        :return: particle_map 基于运动方向的局部地图 Tensor[height,width,channel=3] float32
        """
        old_partcile_states = tf.cast(old_partcile_states,tf.float32)
        new_particle_states = tf.cast(new_particle_states,tf.float32)
        particle_map_length = particle_map_length
        map_shape = tf.shape(map_data)[0:]
        map_shape = tf.cast(map_shape, tf.float32)

        # 计算前后坐标的变化量，并以此计算运动向量的角度temp_theta
        dis = (tf.subtract(new_particle_states[1], old_partcile_states[1]),
               tf.subtract(new_particle_states[0], old_partcile_states[0]))
        temp_theta = tf.to_float(math_ops.atan2(dis[1], dis[0]))

        # 计算围绕在新坐标周围的，没有经过旋转的局部地图的四个顶点坐标
        top_left_point = [new_particle_states[0] - particle_map_length / 2,
                          new_particle_states[1] - particle_map_length / 2]
        top_right_point = [new_particle_states[0] + particle_map_length / 2,
                           new_particle_states[1] - particle_map_length / 2]
        bottom_left_point = [new_particle_states[0] - particle_map_length / 2,
                             new_particle_states[1] + particle_map_length / 2]
        bottom_right_point = [new_particle_states[0] + particle_map_length / 2,
                              new_particle_states[1] + particle_map_length / 2]

        # 计算四个顶点坐标旋转后得到的新坐标
        new_top_left_point = self.getRotatePoint(map_shape, new_particle_states, temp_theta, top_left_point)
        new_top_right_point = self.getRotatePoint(map_shape, new_particle_states, temp_theta, top_right_point)
        new_bottom_left_point = self.getRotatePoint(map_shape, new_particle_states, temp_theta, bottom_left_point)
        new_bottom_right_point = self.getRotatePoint(map_shape, new_particle_states, temp_theta, bottom_right_point)
        new_point = tf.reshape([new_top_left_point, new_top_right_point, new_bottom_left_point, new_bottom_right_point],
                               [4, 2])

        # 计算四个新坐标的横纵坐标之差的极值，该极值构成了能完整圈住旋转后的局部地图的大矩形的shape
        new_shape_x = tf.cast(new_point[tf.argmax(new_point[:, 0], 0), 0] - new_point[tf.argmin(new_point[:, 0], 0), 0],
                              tf.int32)
        new_shape_y = tf.cast(new_point[tf.argmax(new_point[:, 1], 0), 1] - new_point[tf.argmin(new_point[:, 1], 0), 1],
                              tf.int32)

        # 对四个新坐标的合法性进行判断，使其不会越出地图的边界
        new_point = self.pointLegalCheck(map_shape, new_point, [new_shape_x, new_shape_y])

        # 计算该大矩形的左上顶点坐标，依据大矩形的左上顶点坐标切割地图
        new_corner_x = tf.cast(new_point[tf.argmin(new_point[:, 0], 0), 0], tf.int32)
        new_corner_y = tf.cast(new_point[tf.argmin(new_point[:, 1], 0), 1], tf.int32)
        map_cut = tf.image.crop_to_bounding_box(map_data, offset_height=new_corner_y,
                                                offset_width=new_corner_x,
                                                target_height=new_shape_y,
                                                target_width=new_shape_x)

        # 反向旋转切割后的地图，使原本的运动方向与坐标轴方向（Y轴正方向）平行
        map_rotate = image.rotate(map_cut, -1.0 * temp_theta)

        # 计算位于居中的，我们最终需要的particle_map_length大小的局部地图的左上顶点，进行第二次切割
        particle_map_length = tf.to_int32(particle_map_length)
        temp_corner_width = tf.cast(tf.div(tf.subtract(new_shape_x, particle_map_length), 2), tf.int32)
        temp_corner_height = tf.cast(tf.div(tf.subtract(new_shape_y, particle_map_length), 2), tf.int32)
        particle_map = tf.image.crop_to_bounding_box(map_rotate, offset_height=temp_corner_height,
                                                     offset_width=temp_corner_width,
                                                     target_height=particle_map_length,
                                                     target_width=particle_map_length)
        return particle_map

    def getRotatePoint(self, map_shape, rotate_center, rotate_theta, origin_point):
        """
        实现功能，得到绕旋转中心旋转theta角度后的坐标
        :param map_shape:原始地图的尺寸，因为Image中的坐标原点在图片左上角，需要改变坐标系    Tensor-[height,width,channel]
        :param rotate_center:旋转中心   Tensor-[loc_x,loc_y]
        :param rotate_theta:旋转角度   Tensor-[theta]
        :param origin_point:需要进行旋转操作的点集 Tensor-[loc_x,loc_y]
        :return: rotate_point_list: Tensor-[loc_x,loc_y]
        """
        row = map_shape[0]
        center_x = rotate_center[0]
        center_y = row - rotate_center[1]
        point_x = origin_point[0]
        point_y = row - origin_point[1]

        after_rotate_x = math_ops.round(
            (point_x - center_x) * math_ops.cos(rotate_theta) - (point_y - center_y) * math_ops.sin(
                rotate_theta) + center_x)
        after_rotate_y = row - math_ops.round(
            (point_x - center_x) * math_ops.sin(rotate_theta) + (point_y - center_y) * math_ops.cos(
                rotate_theta) + center_y)
        rotate_point = [after_rotate_x, after_rotate_y]
        rotate_point = tf.reshape(rotate_point, [2])
        return rotate_point

    def pointLegalCheck(self, map_shape, point, box_shape):
        """
        检测旋转后的点是否越界
        :param map_shape: 原始地图大小Tensor-[height,width,channel]
        :param point: 旋转后的点集Tensor-[4,2]
        :param box_shape: 裁剪图片区域的大小Tensor-[2]
        :return: 合法性检测后的点集
        """
        x_move_list = []
        y_move_list = []
        map_shape = tf.cast(map_shape, tf.float32)
        # 依次计算四个顶点与地图边界的偏移大小
        for i in range(4):
            x_move = tf.cond(tf.less(point[i][0], 0.0),
                             true_fn=lambda: 0.0 - point[i][0],  # 若坐标小于零，则需要向正方向移动
                             false_fn=lambda: tf.cond(tf.less(map_shape[1], point[i][0]),
                                                      # 若坐标大于地图边界，则需要向负方向移动
                                                      true_fn=lambda: map_shape[1] - point[i][0],
                                                      false_fn=lambda: 0.0  # 坐标在地图中无需移动
                                                      )
                             )
            y_move = tf.cond(tf.less(point[i][1], 0.0),
                             true_fn=lambda: 0.0 - point[i][1],
                             false_fn=lambda: tf.cond(tf.less(map_shape[0], point[i][1]),
                                                      true_fn=lambda: map_shape[0] - point[i][1],
                                                      false_fn=lambda: 0.0
                                                      )
                             )
            x_move_list.append(x_move)
            y_move_list.append(y_move)

        # 依据四个顶点的偏移，得到坐标的修正值
        x_move = self.getPixelMove(tf.reshape(x_move_list, [4]))
        y_move = self.getPixelMove(tf.reshape(y_move_list, [4]))

        # 得到修正后的四个顶点坐标
        point_x = tf.add(point[:, 0], x_move)
        point_y = tf.add(point[:, 1], y_move)
        point = tf.stack([point_x, point_y], 1)
        return point

    def getPixelMove(self, move_list):
        """
        pointLegalCheck的附加函数，返回最大的绝对值位移
        :param move_list: 四个角点与是否越界检测得到的与图片的差值 Tensor-[4]
        :return: max_move 最大位移 Tensor-[1]
        """
        move_max = tf.argmax(move_list)
        move_min = tf.argmin(move_list)
        max_move_index = tf.cond(tf.less(tf.abs(move_min), move_max),
                                 true_fn=lambda: move_max,
                                 false_fn=lambda: move_min
                                 )
        max_move = move_list[max_move_index]
        return max_move

    @staticmethod
    def mapFeatures(particle_maps):
        assert particle_maps.get_shape().as_list()[1:3] == [100, 100]
        data_format = 'channels_last'

        with tf.variable_scope("map"):
            x = particle_maps
            convs = [
                conv2_layer(
                    32, (5, 5), activation=None, padding='same', data_format=data_format,
                    use_bias=True, layer_i=1)(x),
                conv2_layer(
                    64, (5, 5), activation=None, padding='same', data_format=data_format,
                    use_bias=True, layer_i=2)(x),
                conv2_layer(
                    32, (5, 5), activation=None, padding='same', data_format=data_format,
                    use_bias=True, layer_i=3)(x),
            ]
            x = tf.concat(convs, axis=-1)
            x = tf.contrib.layers.layer_norm(x, activation_fn=tf.nn.relu)

            x = tf.layers.max_pooling2d(x, pool_size=(3, 3), strides=(2, 2), padding="same")

            convs = [
                conv2_layer(
                    4, (3, 3), activation=None, padding='same', data_format=data_format,
                    use_bias=True, layer_i=4)(x),
                conv2_layer(
                    4, (5, 5), activation=None, padding='same', data_format=data_format,
                    use_bias=True, layer_i=5)(x),
            ]
            x = tf.concat(convs, axis=-1)
            x = tf.contrib.layers.layer_norm(x, activation_fn=tf.nn.relu)
            x = tf.layers.max_pooling2d(x, pool_size=(3, 3), strides=(2, 2), padding="valid")
            x = dense_layer(1, activation=None, use_bias=True, name='fc1')(x)
        """
        with tf.variable_scope("fc"):

            # pad manually to match different kernel sizes
            x_pad1 = tf.pad(x, paddings=tf.constant([[0, 0], [1, 1, ], [1, 1], [0, 0]]))
            convs = [
                    locallyconn2_layer(
                        4, (5, 5), activation='relu', padding='valid', data_format=data_format,
                        use_bias=True, layer_i=6)(x),
                    locallyconn2_layer(
                        2, (5, 5), activation='relu', padding='valid', data_format=data_format,
                        use_bias=True, layer_i=7)(x_pad1),
            ]
            x = tf.concat(convs, axis=-1)
            x = tf.layers.max_pooling2d(x, pool_size=(3, 3), strides=(2, 2), padding="valid")
            print(x.get_shape().as_list())
            x = dense_layer(1,activation=None,use_bias=True,name='fc1')(x)
            print(x.get_shape().as_list())
        """
        return x

    @staticmethod
    def vectorFeatures(weight_vector):
        with tf.variable_scope("weight_fc"):
            x = weight_vector
            x = dense_layer(1, activation=None, use_bias=True, name='fc2')(x)
        return x

    def weightVariable(layer_shape):
        initial = tf.truncated_normal(layer_shape, stddev=0.1)
        return tf.Variable(initial)

    def biasVariable(layer_shape):
        initial = tf.constant(0.1, shape=layer_shape)
        return tf.Variable(initial)
