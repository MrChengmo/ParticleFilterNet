#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
PFcell

Created on Tue Nov 20 12:50:12 2018

@author: silence
"""
import numpy as np
import tensorflow as tf
import tensorflow.contrib.eager as tfe
from tensorflow.nn.rnn_cell import RNNCell as rnn
from tensorflow.contrib import image
from tensorflow.python.ops import math_ops
tfe.enable_eager_execution()

class PFCell(rnn):
    """
    PF-NET核心模块，使用RNN结构实现粒子滤波过程，RNN的隐状态对应于粒子集
    转移，观测及重采样模型被设计为可微的神经网络单元
    将粒子滤波算法视为计算图或可微分的程序
    Cell inputs: map,ins
    Cell states: particle_states,particle_weights
    Cell outputs: particle_states,particle_weights
    """   
    def __init__(self,map_data,paramters,batch_size,particle_nums):
        super(PFCell,self).__init__()
        self._map = map_data
        self._parameters = paramters
        self._batch_size = batch_size
        self._particle_nums = particle_nums
        self._states_shape = (batch_size,particle_nums,3)
        self._weights_shape = (batch_size,particle_nums,1)
        
    @property
    def state_size(self):
        return (tf.TensorShape(self._states_shape[1:]),tf.TensorShape(self._weights_shape[1:]))
    
    @property
    def output_size(self):
        return (tf.TensorShape(self._states_shape[1:]),tf.TensorShape(self._weights_shape[1:]))

    def __call__(self,inputs,states):
        with tf.variable_scope(tf.get_variable_scope()):
            particle_states,particle_weights = states
            
            #update particle by insX/insY
            new_particle_states = self.transitionModel(particle_states,inputs)
            
            #calc new weights by CNN
            new_particle_weights = self.obervationModel(self._map,particle_states,new_particle_states)
            particle_weights += new_particle_weights
            
            #resample particles
            new_states,new_weights = self.resampleModel(particle_states,particle_weights,
                                            resample_para=self._parameters.resample_para)
        return new_states,new_weights
        
    def transitionModel(self,particle_states,ins):
        """
        particle_states update by ins information and transition noise
        """
        distance_para = self._parameters.transition_para[0]/self._parameters.map_pixel_para
        #rotate_para = self._parameters.transition_para[1]
        with tf.name_scope('transition'):
            loc_x,loc_y = tf.unstack(particle_states,axis=-1,num=2)
            ins_x,ins_y = tf.unstack(ins,axis=-1,num=2)
            
            ins_x += tf.random_normal(ins_x.get_shape(),mean=0.0,stddev=1.0)*distance_para
            ins_y += tf.random_normal(ins_y.get_shape(),mean=0.0,stddev=1.0)*distance_para
            
            new_particle_states = tf.stack([loc_x+ins_x,loc_y+ins_y],axis=-1)
        return new_particle_states
        
    def obervationModel(self,map_data,old_particle_states,new_particle_states):
        """
        this model transform the big map to particle maps for each particle,where a particle map 
        is a local view from the state defined by the particle.And we use Cnn to calc the particle
        weight by particle maps
        """
        
        particle_maps = self.mapTransform(self._map,old_particle_states,new_particle_states,
                                   self._parameters.particle_map_shape)
        particle_maps = tf.reshape(particle_maps,[self._batch_size * self._particle_nums]
                                    +particle_maps.shape.as_list()[2:])
        map_features = self.mapFeatures(particle_maps)
        
        new_particle_weights = tf.reshape(map_features,[self._batch_size,self._particle_nums])
        return new_particle_weights
    
    @staticmethod    
    def resampleModel(particle_states,particle_weights,resample_para):
        with tf.name_scope('resample'):
            assert 0.0 < resample_para <= 1.0
            batch_size, num_particles = particle_states.get_shape().as_list()[:2]

            # normalize
            particle_weights = particle_weights - tf.reduce_logsumexp(particle_weights, 
                                                                      axis=-1, keep_dims=True)

            uniform_weights = tf.constant(-np.log(num_particles), 
                                          shape=(batch_size, num_particles), dtype=tf.float32)

            # build sampling distribution, q(s), and update particle weights
            if resample_para < 1.0:
                # soft resampling
                q_weights = tf.stack([particle_weights + np.log(resample_para), 
                                      uniform_weights + np.log(1.0-resample_para)], axis=-1)
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
            helper = tf.range(0, batch_size*num_particles, delta=num_particles, dtype=tf.int32)  # (batch, )
            indices = indices + tf.expand_dims(helper, axis=1)

            particle_states = tf.reshape(particle_states, (batch_size * num_particles, 3))
            particle_states = tf.gather(particle_states, indices=indices, axis=0)  # (batch_size, num_particles, 3)

            particle_weights = tf.reshape(particle_weights, (batch_size * num_particles, ))
            particle_weights = tf.gather(particle_weights, indices=indices, axis=0)  # (batch_size, num_particles,)

        return particle_states, particle_weights
    
    def mapTransform(self,map_data,old_particle_states,new_particle_states):
        """
        :map_data: tf op Big IndoorMap
        :param old_particle_states: tf op (batch, K, 2), time t-1 's particle states
        :param new_particle_states: tf op (batch, K, 2), time   t 's transition particle states
        """
        batch_size, num_particles = old_particle_states.get_shape().as_list()[:2]
        total_samples = batch_size * num_particles
        old_flat_states = tf.reshape(old_particle_states,[total_samples,2])
        new_flat_states = tf.reshape(new_particle_states,[total_samples,2])
        particle_map_list = []
        for i in range(total_samples):
            particle_map_list.append(self.mapCut(map_data,old_flat_states[i],new_flat_states[i]))
        particle_map = tf.reshape(particle_map_list,(batch_size,num_particles,
                            self._parameters.particle_map_shape[0],self._parameters.particle_map_shape[1],
                            map_data.shape.as_list()[-1]))
        return particle_map
        
    
    def mapCut(self,map_data,old_partcile_states,new_particle_states):
        angle_64 = math_ops.atan2(new_particle_states[1]-old_partcile_states[1],
                                  new_particle_states[0]-old_partcile_states[0])
        angle_32 = tf.to_float(angle_64)
        corner = math_ops.sub(new_particle_states,self._parameters.particle_map_length)
        top_left_corner = tf.to_int32(corner,name='ToInt32')
        map_cut = tf.image.crop_to_bounding_box(map_data,offset_height=top_left_corner[0],
                    offset_width=top_left_corner[1],target_height=self._parameters.particle_map_length,
                    target_width= self._parameters.particle_map_length)
        map_rotate = image.rotate(map_cut,angle_32)
        map_rotate = tf.reshape(map_rotate,(self._parameters.particle_map_shape[0],
                                self._parameters.particle_map_shape[1],map_data.shape.as_list()[-1]))
        return map_rotate
        
    @staticmethod
    def mapFeatures(particle_maps):

        return particle_weights
        
        
        
        
        
        
        
        
        
        
        
        
