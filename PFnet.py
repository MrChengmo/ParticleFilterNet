#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
PF-net

Created on Wed Nov 21 19:29:10 2018

@author: silence
"""
import numpy as np
import tensorflow as tf
import tensorflow.contrib.eager as tfe
from PFcell import PFCellClass


class PFnetClass(object):
    """
    Build PF-net by PFCell ,defines losses and training ops 
    """

    def __init__(self, map_data, inputs, labels, parameters, is_training):
        """
        creat all tf ops for PF-net
        :parm map_data:indoor location map
        :parm inputs:list of tf ops,have following elements:init_particle_states
                    ins_information,is_first_step
        :parm labels:tf op,labels for training,the true states along the trajectory
        :parm parameters:parsed arguement
        :parm is_training
        """
        self._parameters = parameters
        self._output = []
        self._hidden_states = []
        self._train_loss = None
        self._valid_loss = None
        self._all_distance = None
        self._global_step = None
        self._learning_rate = None
        self._train = None
        self._update_state = tf.constant(0)
        self._particle_nums = parameters.particle_nums
        init_states = self.initParticleStates(labels[:,1,:])
        self.build(map_data=map_data, ins=inputs[:, 1:, :], init_particle_states=init_states,
                   labels=labels[:, 1:, :], is_training=is_training)

    def build(self, map_data, ins, init_particle_states, labels, is_training):
        self.outputs = self.buildRnn(map_data, ins, init_particle_states)
        self.buildLoss(self.outputs, true_states=labels)
        if is_training:
            self.buildTrain()

    def initParticleStates(self,init_loc):
        batch_size = init_loc.get_shape().as_list()[0]
        init_states = []
        for i in range(batch_size):
            init_batch_states = tf.tile([init_loc[i]],multiples=[self._particle_nums,1 ])
            init_states.append(init_batch_states)
        init_states = tf.convert_to_tensor(init_states)
        return init_states


    def saveState(self, sess):
        return sess.run(self.hidden_states)

    def loadState(self, sess, savedState):
        return sess.run(self.hidden_sates,
                        feed={self.hidden_states[i]:
                                  savedState[i] for i in range(len(self.hidden_states))})

    def buildLoss(self, particle_states, particle_weights, true_states):
        lin_weights = tf.nn.softmax(particle_weights, dim=-1)
        true_coords = true_states[:, :, :2]
        mean_coords = tf.reduce_mean(tf.multiply(
            particle_states[:, :, :, :2], lin_weights[:, :, :, None]), axis=2)
        coord_diffs = mean_coords - true_coords
        loss_coords = tf.reduce_sum(tf.square(coord_diffs), axis=2)

        loss_pred = tf.reduce_mean(loss_coords, name='prediction_loss')
        loss_reg = tf.multiply(tf.losses.get_regularization_loss(),
                               self._parameters.l2scale, name='L2')
        loss_total = tf.add_n([loss_pred, loss_reg], name="training_loss")
        self._all_distance2 = loss_coords
        self._valid_loss = loss_pred
        self._train_loss = loss_total
        return loss_total

    def buildTrain(self):
        assert self._train is None and self._global_step is None and self._learning_rate is None

        with tf.device("/cpu:0"):
            self._global_step = tf.get_variable(
                initializer=tf.constant_initializer(0.0), shape=(), trainable=False, name='global_step')
            self._learning_rate = tf.train.exponential_decay(
                self._parameters.learning_rate, self._global_step, decay_steps=1,
                decay_rate=self._parameters.decay_rate, staircase=True, name="learning_rate")

            optimizer = tf.train.RMSPropOptimizer(self._learning_rate, decay=0.9)
        with tf.control_dependencies(tf.get_collection(tf.GraphKeys.UPDATE_OPS)):
            self._train = optimizer.minimize(self._train_loss, global_step=None,
                                             var_list=tf.trainable_variables())
        return self._train

    def buildRnn(self, map_data, ins, init_particle_states):
        batch_size= ins.get_shape().as_list()[0]
        particle_nums = self._particle_nums
        map_shape = tf.shape(map_data).eval()

        init_particle_weights = tf.constant(np.log(1.0 / float(particle_nums)),
                                            shape=(batch_size, particle_nums), dtype=tf.float64)
        assert len(self._hidden_states) == 0
        self._hidden_states = [
            tf.get_variable("particle_states", shape=init_particle_states.get_shape(),
                            dtype=init_particle_states.dtype, initializer=tf.constant_initializer(0), trainable=False),
            tf.get_variable("particle_weights", shape=init_particle_weights.get_shape(),
                            dtype=init_particle_weights.dtype, initializer=tf.constant_initializer(0), trainable=False),
        ]

        state = (init_particle_states, init_particle_weights)
        with tf.variable_scope("rnn"):
            init_cell = PFCellClass(map_data=tf.zeros(map_shape, dtype=tf.float64),
                                    paramters=self._parameters, batch_size=1, particle_nums=1)
            init_cell(tf.zeros([1, 2], dtype=np.float64),  # inputs
                      (tf.zeros([1, 1, 2], dtype=np.float64),
                       tf.zeros([1, 1], dtype=np.float64))  # state
                      )

            tf.get_variable_scope().reuse_variables()
            cell_func = PFCellClass(map_data=map_data, paramters=self._parameters,
                                    batch_size=batch_size, particle_nums=particle_nums)

            outputs, states = tf.nn.dynamic_rnn(cell=cell_func,
                                                inputs=ins,
                                                initial_state=state,
                                                swap_memory=True,
                                                time_major=False,
                                                parallel_iterations=1,
                                                scope=tf.get_variable_scope())
        particle_states, particle_weights = outputs
        with tf.control_dependencies([particle_states, particle_weights]):
            self._update_state = tf.group(
                *(self._hidden_states[i].assign(state[i]) for i in range(len(self._hidden_states))))

        return particle_states, particle_weights
