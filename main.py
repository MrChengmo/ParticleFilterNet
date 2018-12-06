#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
PF main

Created on Thu Nov 15 15:37:26 2018

@author: Chengmo
"""
import numpy as np
import tensorflow as tf
# tf.enable_eager_execution()
import matplotlib.pyplot as plt
from argument import parse_args
from DataProcess import LabelData, MapData
from PFnet import PFnetClass
import math

"""使用Tensorflow的eager模式方便调试"""


def run_training(params):
    """ Run training with the parsed arguments """
    with tf.Graph().as_default():
        # 各个数据读取类初始化
        trainData = LabelData(params.train_files_path, params.train_ration, params.read_all)
        testData = LabelData(params.test_files_path, params.train_ration, params.read_all)
        mapData = MapData(params.map_files_path)

        num_train_samples = trainData.getBatchNums(params.time_step) / params.batchsize
        num_train_samples = math.floor(num_train_samples)
        train_data = trainData.getData(params.epochs)
        train_data = train_data.batch(params.batchsize, drop_remainder=True)
        train_iter = train_data.make_one_shot_iterator()
        inputs = train_iter.get_next()

        num_test_samples = testData.getBatchNums(params.time_step)
        test_data = testData.getData(params.epochs)
        test_data = test_data.batch(params.batchsize, drop_remainder=True)
        test_iter = test_data.make_one_shot_iterator()
        test_inputs = test_iter.get_next()

        map_data = mapData.getMap()

        if params.seed is not None:
            tf.set_random_seed(params.seed)

        # training data and network
        with tf.variable_scope(tf.get_variable_scope(), reuse=False):
            train_brain = PFnetClass(map_data, inputs=inputs[0], labels=inputs[1], parameters=params,is_training=True)
        print("successful train_op")
        # test data and network
        with tf.variable_scope(tf.get_variable_scope(), reuse=True):
            test_brain = PFnetClass(map_data, inputs=test_inputs[0], labels=test_inputs[1], parameters=params,is_training=False)
        print("successful test_op")
        # Add the variable initializer Op.
        init_op = tf.group(tf.global_variables_initializer(), tf.local_variables_initializer())

        # training session


        with tf.Session() as sess:
            sess.run(init_op)
            print("successful init_op")

            try:
                decay_step = 0

                # repeat for a fixed number of epochs
                for epoch_i in range(params.epochs):
                    epoch_loss = 0.0
                    periodic_loss = 0.0
                    print(num_train_samples)
                    print("successful epoch")
                    # run training over all samples in an epoch
                    for step_i in range(num_train_samples):
                        _, loss, _ = sess.run([train_brain._train_op, train_brain._train_loss_op,
                                               train_brain._update_state_op])
                        periodic_loss += loss
                        epoch_loss += loss
                        print(step_i)
                        # print accumulated loss after every few hundred steps


                    # print the avarage loss over the epoch
                    print(epoch_loss)

                    # run validation
                    #validation(sess, test_brain, num_samples=num_test_samples, params=params)

                    #  decay learning rate
                    """
                    if epoch_i + 1 % params.decaystep == 0:
                        decay_step += 1
                        current_learning_rate = sess.run(train_brain._learning_rate_op)
                        tqdm.tqdm.write("Decreased learning rate to %f." % (current_learning_rate))
                    """
            except KeyboardInterrupt:
                pass

            except tf.errors.OutOfRangeError:
                print("data exhausted")


        print("Training done. Model is saved to %s" % (params.logpath))


def validation(sess, brain, num_samples, params):
    """
    Run validation
    :param sess: tensorflow session
    :param brain: network object that provides loss and update ops, and functions to save and restore the hidden state.
    :param num_samples: int, number of samples in the validation set.
    :param params: parsed arguments
    :return: validation loss, averaged over the validation set
    """

    fix_seed = (params.validseed is not None and params.validseed >= 0)
    if fix_seed:
        np_random_state = np.random.get_state()
        np.random.seed(params.validseed)
        tf.set_random_seed(params.validseed)

    saved_state = brain.save_state(sess)

    total_loss = 0.0
    try:
        for eval_i in range(num_samples):
            loss, _ = sess.run([brain.valid_loss_op, brain.update_state_op])
            total_loss += loss

        print("Validation loss = %f" % (total_loss / num_samples))

    except tf.errors.OutOfRangeError:
        print("No more samples for evaluation. This should not happen")
        raise

    brain.load_state(sess, saved_state)

    # restore seed
    if fix_seed:
        np.random.set_state(np_random_state)
        tf.set_random_seed(np.random.randint(999999))  # cannot save tf seed, so generate random one from numpy

    return total_loss


if __name__ == '__main__':
    params = parse_args()
    run_training(params)
