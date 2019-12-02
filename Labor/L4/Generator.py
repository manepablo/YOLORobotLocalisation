# -*- coding: utf-8 -*-
"""
Created on Mon Nov 18 15:21:39 2019

@author: Paul
"""
from keras.utils import Sequence
import numpy as np
import cv2
import os 

class generator(Sequence):
    def __init__(self, x_path, y_path, batchsize, scale = (1/255)):
        self.x_path = x_path
        self.y_path = y_path
        self.x_data_list = os.listdir(x_path)
        self.y_data_list = os.listdir(y_path)
        self.batchsize = batchsize
        self.scale = scale   
        
    def __len__(self):
        return ((int)(len(self.x_data_list)/self.batchsize))
    
    def __getitem__(self, idx):        
        X = np.zeros(shape=(self.batchsize,256,256,3))
        Y = np.zeros(shape=(self.batchsize,256,256,1))
        try:
            for i in range(0, self.batchsize):
                current_idx=self.batchsize*idx+i;
                X_buff = cv2.imread(self.x_path + '/' + self.x_data_list[current_idx])
                Y_buff = cv2.imread(self.y_path + '/' + self.y_data_list[current_idx], 0)
                Y_buff = np.expand_dims(Y_buff, axis = -1)
                # print(str(X_buff.shape) +'\n'+ str(Y_buff.shape) + '\n' + str(Y[i].shape))  
                X[i] = cv2.resize(X_buff, (256,256), interpolation = cv2.INTER_AREA)
                Y[i] = np.expand_dims(cv2.resize(Y_buff, (256,256), interpolation = cv2.INTER_AREA),3)
                
        except IndentationError:
            print('index Error at: ' + str(current_idx))
            pass
        return X,Y            

