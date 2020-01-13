# -*- coding: utf-8 -*-
"""
Created on Mon Jan 13 13:57:51 2020

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


# load Data
ld = loadData(dimx = x_dim, dimy = y_dim)
if bb.upper() == "2D":    
    xTrain, xTest , yTrain, yTest = ld.LoadAndSafe2D(split_size=split, relativLabels = relLabels)
if bb.upper() == "3D":    
    xTrain, xTest , yTrain, yTest = ld.LoadAndSafe3D(split_size=split, relativLabels = relLabels)

# create the model and show its architecture
if bb.upper() == "2D": 
    localizer = ObjectLocalizer2D( input_shape=(ld.dimy , ld.dimx, 3 ) )
else:
    localizer = ObjectLocalizer3D( input_shape=(ld.dimy , ld.dimx, 3 ) )
model = localizer.get_model()
plot_model(model, show_shapes = True, rankdir = 'TB', expand_nested = True)

localizer.load_model( bb + '_' + 'model.h5' )

for i in range(50,80):
    plotIM_BB(image=xTest[i], boudingBox = yTest[i], boudingBox2 = localizer.predict(np.expand_dims(xTest[i],0))[0], bbType = bb, loss = localizer.iou_metric(yTest[i], localizer.predict(np.expand_dims(xTest[i],0))[0] ))
    
