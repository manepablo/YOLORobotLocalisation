# -*- coding: utf-8 -*-
"""
Created on Mon Dec 16 15:50:14 2019

@author: Paul
"""



from object_localizer import ObjectLocalizer2D, ObjectLocalizer3D
from loadData import loadData 
from PIL import Image , ImageDraw
import numpy as np
from plot_img import plotIM_BB
from tensorflow.keras.utils import plot_model

# Define some parameters befor Training:
# define if bb points are relativ, ranging from 0 to 1 or are absolut pixel values
relLabels = True
# define desired Image size 
x_dim = 512; y_dim = 512
# define splitsize for Test and Train data
split = 0.1
# define if 3d bb or 2d bb type
bb = "3d"
# defin epochs to train
epochs = 15
# defin batch size
batch_size = 1

# load Data
ld = loadData(dimx = x_dim, dimy = y_dim)
if bb.upper() == "2D":    
    xTrain, xTest , yTrain, yTest = ld.LoadAndSafe2D(split_size=split, relativLabels = relLabels)
if bb.upper() == "3D":    
    xTrain, xTest , yTrain, yTest = ld.LoadAndSafe3D(split_size=split, relativLabels = relLabels)

# show some bounding boxes
i = 34
plotIM_BB(image=xTrain[i], boudingBox = yTrain[i], bbType = bb, relativLabels = relLabels)

# print the shape of the data
print( xTrain.shape )
print( yTrain.shape )
print( xTest.shape )
print( yTest.shape )

# create the model and show its architecture
if bb.upper() == "2D": 
    localizer = ObjectLocalizer2D( input_shape=(ld.dimy , ld.dimx, 3 ) )
else:
    localizer = ObjectLocalizer3D( input_shape=(ld.dimy , ld.dimx, 3 ) )
model = localizer.get_model()
plot_model(model, show_shapes = True, rankdir = 'TB', expand_nested = True)


# define the parameter for the training
parameters = {
    'batch_size' : batch_size ,
    'epochs' : epochs ,
    'callbacks' : None ,
    'val_data' : ( xTest , yTest )
}

# train the network
localizer.fit( xTrain , yTrain  , hyperparameters = parameters )
# save the network
localizer.save_model( 'model.h5')

# show some predicted bounding boxes

i = 33
plotIM_BB(image=xTest[i], boudingBox = yTest[i], boudingBox2 = localizer.predict(np.expand_dims(xTest[i],0))[0], bbType = bb)

for i in range(60,80):
    plotIM_BB(image=xTest[i], boudingBox = yTest[i], boudingBox2 = localizer.predict(np.expand_dims(xTest[i],0))[0], bbType = bb)
