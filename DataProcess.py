#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
DataProcess

Created on Thu Nov 15 15:35:36 2018

@author: Chengmo
"""
import os
import math
import tensorflow as tf
import tensorflow.contrib.eager as tfe
import numpy as np
#tfe.enable_eager_execution()

class LabelData():
    """读取训练数据"""
    def __init__(self,path,train_ration = 0.9,read_all = False):
        self._origin_path = path;
        self._train_ration = train_ration;
        self._read_all = read_all
        self._first_read = True  
        self._record_default = [[1.],[1.],[1.],[1.],[1.],[1.]]
        
    def showName(self,path=''):
        """
        获取path路径下的所有文件及文件夹
        """
        cur_path = path;
        cur_list = os.listdir(cur_path);
        list_dir = [];
        list_file = [];
        for temp in cur_list:
            if os.path.isdir(cur_path + "/" + temp):
                list_dir.append(cur_path + "/" + temp)
            else:
                list_file.append(cur_path + "/" + temp)
                
        return list_dir,list_file
    
    def readData(self,path=''):
        """  
        在path下读取文件，根据self._read_all属性决定是否递归读取所有子文件夹
        """
        list_dir,list_file = self.showName(path)
        new_list_dir = []
        if(self._read_all):
            for floder in list_dir:
                temp_list_dir,temp_list_file = self.showName(floder)
                list_file = list_file + temp_list_file
                new_list_dir = new_list_dir + temp_list_dir
            for floder in new_list_dir:
                list_file = list_file + self.readData(floder)
        return list_file
    
    def getBatchNums(self,time_step):
        """
        根据BATCH的大小得到共有多少组Batch，同时该函数作为数据读取类的入口
        """
        self._time_step = time_step
        self._file_list = self.readData(self._origin_path)
        self._batch_num = 0;
        legal_file_list = []
        
        for file in self._file_list:
            f = open(file,"r")
            row_num = f.readlines()
            "此处将不符合规定的数据文件删除(不予考虑)"
            if(len(row_num)<time_step):
                continue
            else:
                legal_file_list.append(file)
                self._batch_num += math.ceil(len(row_num)/time_step)
        
        self._file_list = legal_file_list
        return self._batch_num
    
    """读取数据有两种主要的方式，先用Np读取处理再转为DataSet，或者直接用TF-CSVDATASET读取"""            
    def getData(self,num_epochs = 1):
        data = np.zeros((1,6))
        for file in self._file_list:
            content = np.loadtxt(file,dtype = np.float64,delimiter = ',')
            length = len(content)
            #将惯导数据从自有（3933）坐标系转换为相对变化量
            for i in range(length-1,-1,-1):
                if i==0:
                    content[i,0:3]=0
                else:
                    content[i,0:3] = content[i,0:3]-content[i-1,0:3]
            if length % self._time_step == 0:
                data = np.vstack((data,content))
            else:
                """
                1-content中删除大于整除部分的行
                2-从数据中补齐最后的一组batch
                3-将补齐的数据加入data
                """
                extra_line_num = length % self._time_step
                extra_data = content[length-self._time_step:length]
                content = content[:length-extra_line_num]
                new_data = np.vstack((content,extra_data))
                data = np.vstack((data,new_data))
        data = np.delete(data,0,axis = 0)
        features = data[:,0:2]
        labels = data[:,3:5]
        dataset = tf.data.Dataset.from_tensor_slices((features,labels)).batch(self._time_step,drop_remainder=True)
        dataset = dataset.repeat(num_epochs)
        return dataset
    

class MapData:
    """读取地图数据，并为未来复杂地图操作留下接口"""
    def __init__(self,path):
        self._origin_path = path
        
    def getMap(self):
        file = tf.read_file(self._origin_path)
        self._image = tf.image.decode_image(file,channels=3)
        self._image = tf.cast(self._image, tf.float64)
        return self._image
#-----------------------------------------------------------------------------"        
"""
LabelData单元测试模块，测试内容：
1-单文件是否能够正确读取 OK 
2-根文件夹下多个文件能否正确计算num OK
3-根文件夹下有多个子文件夹，能否全部读取 OK
4-根文件夹下有多个深层子文件夹，能否正确读取 OK
5-读取数据代码能否正常工作 OK
6-数据Batch划分能否正常工作 OK
"""   
"""
DATA_PATH = '/home/silence/PF/data' 
BATCH_SIZE = 1
TIME_STEP = 10
TRAIN_DATA_RATIO = 0.9
READ_ALL = True
EPOCHS = 1
dataRead = LabelData(DATA_PATH,TRAIN_DATA_RATIO,READ_ALL) 
each_loop_num = dataRead.getBatchNums(TIME_STEP)
print(each_loop_num)
dataset = dataRead.getData(num_epochs= EPOCHS)  
dataset = dataset.batch(5) 
print(dataset.output_shapes)
print(dataset.output_types)
iterator = dataset.make_one_shot_iterator()
one_element = iterator.get_next()
with tf.Session() as sess:
    for i in range(1):
        print(sess.run(one_element))
"""
"""
path ='/home/silence/PF/map.jpg'
mapRead = MapData(path)
map_data = mapRead.getMap()
with tf.Session() as sess:

    imsize = tf.shape(map_data)
    print(imsize.eval())
"""


#-----------------------------------------------------------------------------"      
        

        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
                
                
                