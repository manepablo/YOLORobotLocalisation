# -*- coding: utf-8 -*-
"""
Created on Mon Dec 16 15:50:14 2019

@author: Paul
"""



from object_localizer import ObjectLocalizer2D, ObjectLocalizer3D
from loadData import loadData 
from PIL import Image , ImageDraw
import numpy as np

ld = loadData(dimx= 512, dimy = 512)

input_dim = 228

xTrain, xTest , yTrain, yTest = ld.LoadAndSafe3D(split_size=0.1)

print( xTrain.shape )
print( yTrain.shape )
print( xTest.shape )
print( yTest.shape )

localizer = ObjectLocalizer3D( input_shape=(ld.dimy , ld.dimx, 3 ) )




parameters = {
    'batch_size' : 1 ,
    'epochs' : 10 ,
    'callbacks' : None ,
    'val_data' : ( xTest , yTest )
}

localizer.fit( xTrain , yTrain  , hyperparameters=parameters )
localizer.save_model( 'model.h5')