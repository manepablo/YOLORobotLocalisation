# -*- coding: utf-8 -*-
"""
Created on Mon Nov 18 15:21:39 2019

@author: Paul Manea und Tim Eisenacher
"""
import os 
import json
import numpy as np
import cv2
from sklearn.model_selection import train_test_split

class loadData():
    def __init__(self, xPath = os.path.dirname(os.path.abspath(__file__)) + '\\pictures', yPath = os.path.dirname(os.path.abspath(__file__)) + '\\labels', dimx= 640, dimy = 480):
        self.xPath = xPath
        self.yPath = yPath
        self.dimx = dimx
        self.dimy = dimy
        self.dir = os.path.dirname(os.path.abspath(__file__))
        
    def loadImg(self):
       if not os.path.isdir(self.xPath):
           print(self.xPath + 'is no Path')
           return
       x_data_list = os.listdir(self.xPath)
       images = []
       
       for pics in x_data_list:           
           image = cv2.imread(self.xPath + '/' + pics)
           images.append( np.asarray( image ) / 255 )
       return images
   
    def loadLabel(self):
       if not os.path.isdir(self.yPath):
           print(self.yPath + 'is no Path')
           return
       data = []
       y_data_list = os.listdir(self.yPath)
       for label in y_data_list: 
           json_file = open(self.yPath + '\\' + label)
           data.append(json.loads(json_file.read()))    
       return data 
   
    def loadLabel2D(self):
        data = self.loadLabel()
        labeldata = []
        for dic in data:
            bb = [0.0]*5
#            bb[0] = int(dic['2dBoundingBox'][0]*self.dimx)
#            bb[1] = int(dic['2dBoundingBox'][1]*self.dimy)
#            bb[2] = int(dic['2dBoundingBox'][2]*self.dimx)
#            bb[3] = int(dic['2dBoundingBox'][3]*self.dimy)  
            bb[0] = dic['2dBoundingBox'][0]
            bb[1] = dic['2dBoundingBox'][1]
            bb[2] = dic['2dBoundingBox'][2]
            bb[3] = dic['2dBoundingBox'][3]  
            bb[4] = int(1)
            labeldata.append(bb)
        
        return np.array(labeldata)
        

    def LoadAndSafe2D(self, split_size=0.5, save = False):
        train_features , test_features ,train_labels, test_labels = train_test_split(self.loadImg() , self.loadLabel2D() , test_size = split_size)
        if save:
            np.save( os.path.join( self.dir , 'train_x.npy' ) , train_features )
            np.save( os.path.join( self.dir , 'train_y.npy' )  , train_labels )
            np.save( os.path.join( self.dir , 'test_x.npy' ) , test_features )
            np.save( os.path.join( self.dir , 'test_y.npy' ) , test_labels ) 
        return np.array(train_features) , np.array(test_features) , np.array(train_labels), np.array(test_labels)
        
    def loadLabel3D(self):
        data = self.loadLabel()
        labeldata = []

        for dic in data:
            bb = [0.0]*17
            bb[0] = int(dic['3dBoundingBox'][0]*self.dimx)
            bb[1] = int(dic['3dBoundingBox'][1]*self.dimy)
            bb[2] = int(dic['3dBoundingBox'][2]*self.dimx)        
            bb[3] = int(dic['3dBoundingBox'][3]*self.dimy)
            bb[4] = int(dic['3dBoundingBox'][4]*self.dimx)  
            bb[5] = int(dic['3dBoundingBox'][5]*self.dimy)
            bb[6] = int(dic['3dBoundingBox'][6]*self.dimx)        
            bb[7] = int(dic['3dBoundingBox'][7]*self.dimy)
            bb[8] = int(dic['3dBoundingBox'][8]*self.dimx)
            bb[9] = int(dic['3dBoundingBox'][9]*self.dimy)
            bb[10] = int(dic['3dBoundingBox'][10]*self.dimx)        
            bb[11] = int(dic['3dBoundingBox'][11]*self.dimy)
            bb[12] = int(dic['3dBoundingBox'][12]*self.dimx)  
            bb[13] = int(dic['3dBoundingBox'][13]*self.dimy)
            bb[14] = int(dic['3dBoundingBox'][14]*self.dimx)        
            bb[15] = int(dic['3dBoundingBox'][15]*self.dimy)            
            bb[16] = int(1)
            labeldata.append(bb)
        return np.array(labeldata)
        
    def LoadAndSafe3D(self, split_size=0.5, save = False):
        train_features , test_features ,train_labels, test_labels = train_test_split(self.loadImg() , self.loadLabel3D() , test_size = split_size)
        if save:
            np.save( os.path.join( self.dir , 'train_x.npy' ) , train_features )
            np.save( os.path.join( self.dir , 'train_y.npy' )  , train_labels )
            np.save( os.path.join( self.dir , 'test_x.npy' ) , test_features )
            np.save( os.path.join( self.dir , 'test_y.npy' ) , test_labels ) 
        return np.array(train_features) , np.array(test_features) , np.array(train_labels), np.array(test_labels)    
            
           
ld = loadData()
