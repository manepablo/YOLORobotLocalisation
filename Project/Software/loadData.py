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
            bb = [0.0]*4
            bb[0] = int(dic['2dBoundingBox'][0]*self.dimx)
            bb[1] = int(dic['2dBoundingBox'][1]*self.dimy)
            bb[2] = int(dic['2dBoundingBox'][2]*self.dimx)
            bb[3] = int(dic['2dBoundingBox'][3]*self.dimy)             
            labeldata.append(bb)
        return labeldata
    
    #------bearbeiten-----
    def splitandsafe(self, test_size=0.5):
        train_features , test_features ,train_labels, test_labels = train_test_split(self.loadImg() , self.loadLabel2D() , test_size)
        np.save( os.path.join( self.dir , 'x.npy' ) , train_features )
        np.save( os.path.join( self.dir , 'y.npy' )  , train_labels )
        np.save( os.path.join( self.dir , 'test_x.npy' ) , test_features )
        np.save( os.path.join( self.dir , 'test_y.npy' ) , test_labels )
            
            
           
ld = loadData()
