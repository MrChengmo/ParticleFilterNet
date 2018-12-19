# -*- coding: utf-8 -*-
"""
PFnet parameter

Created on Mon Nov 26 15:05:31 2018

@author: Silence
"""
import configargparse
import numpy as np


def parse_args(args=None):
    "--------------------------------超参数-------------------------------------"
    p = configargparse.ArgParser(default_config_files=[])


    p.add('--train_files_path', default='train_data/',
          help='Data file(s) for training (csv).')
    p.add('--test_files_path', default='test_data/',
          help='Data file(s) for validation or evaluation (csv).')
    p.add('--map_files_path', default='map/map.jpg',
          help='Map file(s) for training and testing')

    p.add('--read_all', type=bool, default='True', help='read all folder and file in root path')
    p.add('--train_ration', type=float, default=0.9, help='the ratio of files in all file which used to train model')
    # input configuration
    p.add('--pixel_meter_para', type=float, default=0.04017182, help='单位像素距离对应的实际距离（m）')
    p.add('--meter_pixel_para', type=float, default=24.89307181, help='map_pixel_para的倒数，实际距离一米在地图中为多少像素')
    p.add('--particle_map_shape', nargs='*', default=[100, 100], help='the shape of particle map')
    p.add('--particle_map_length', type=float, default=100.0, help='the length of particle map')

    # PF-net configuration
    p.add('--particle_nums', type=int, default=30, help='Number of particles in PF-net.')
    p.add('--step_stddev', type=float, default=0.8, help='每行进一步时粒子的分散高斯分布的标准差，人均步长为0.8米')

    p.add('--resample', type=str, default='true',
          help='Resample particles in PF-net. Possible values: true / false.')
    p.add('--resample_para', type=float, default=1.0,
          help='Trade-off parameter for soft-resampling in PF-net. Only effective if resample == true. '
               'Assumes values 0.0 < alpha <= 1.0. Alpha equal to 1.0 corresponds to hard-resampling.')
    p.add('--transition_para', nargs='*', default=["1.0", "1.0"],
          help='Standard deviations for transition model. Expects two float values: ' +
               'translation std (meters), rotatation std (radians). Defaults to zeros.')

    # training configuration

    p.add('--batchsize', type=int, default=1, help='Minibatch size for training. Must be 1 for evaluation.')
    p.add('--time_step', type=int, default=16, help='Number of foot step which one traj has.')
    p.add('--learning_rate', type=float, default=0.0001, help='Initial learning rate for training.')
    p.add('--l2scale', type=float, default=0.0, help='Scaling term for the L2 regularization loss.')
    p.add('--epochs', metavar='epochs', type=int, default=10, help='Number of epochs for training.')
    p.add('--decay_step', type=int, default=4, help='Decay the learning rate after every N epochs.')
    p.add('--decay_rate', type=float, default=0.001, help='Rate of decaying the learning rate.')


    p.add('--load', type=str, default="", help='Load a previously trained model from a checkpoint file.')
    p.add('--logpath', type=str, default='log/',
          help='Specify path for logs. Makes a new directory under ./log/ if empty (default).')
    p.add('--seed', type=int, help='Fix the random seed of numpy and tensorflow if set to larger than zero.')
    p.add('--validseed', type=int,
          help='Fix the random seed for validation if set to larger than zero. ' +
               'Useful to evaluate with a fixed set of initial particles, which reduces the validation error variance.')
    p.add('--gpu', type=int, default=0, help='Select a gpu on a multi-gpu machine. Defaults to zero.')


    p.add('--res_path',type=str,default='res_path/',help='记录预测轨迹与真实轨迹，便于观察效果')

    params = p.parse_args(args=args)

    # fix numpy seed if needed
    if params.seed is not None and params.seed >= 0:
        np.random.seed(params.seed)

    # convert multi-input fileds to numpy arrays
    params.transition_para = np.array(params.transition_para, np.float32)

    # convert boolean fields
    if params.resample not in ['false', 'true']:
        print("The value of resample must be either 'false' or 'true'")
        raise ValueError
    params.resample = (params.resample == 'true')

    return params
