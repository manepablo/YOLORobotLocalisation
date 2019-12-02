# -*- coding: utf-8 -*-
"""
Created on Mon Nov 25 10:26:44 2019

@author: Paul
"""
from keras.layers import BatchNormalization, Activation, Input, Dense, Conv2D, MaxPooling2D, UpSampling2D, Conv2DTranspose, concatenate
from keras.models import Model
from keras.callbacks import EarlyStopping
from keras.optimizers import Adam
import matplotlib.pyplot as plt
import numpy as np
from Unet import getUNet
from Generator import generator

# create UNet Modell
model = getUNet((256,256,3))

#optimizer_ = Adam(learning_rate= 0.5);

# compile Model
model.compile(loss='binary_crossentropy', optimizer="adam", metrics=['accuracy'])

# instance Generator Class
gen_train = generator('C:/Users/Paul/Desktop/DLM/images', 'C:/Users/Paul/Desktop/DLM/Masks', 8, scale = (1/255))
gen_test = generator('C:/Users/Paul/Desktop/DLM/imagesValidation', 'C:/Users/Paul/Desktop/DLM/MasksValidation', 8, scale = (1/255))

#
es_cb = EarlyStopping(monitor='val_loss', patience=4, verbose=1, mode='auto')

# Train The Model
model.fit_generator(gen_train, epochs=15, verbose=1,   shuffle=True, initial_epoch=0, callbacks = [es_cb])


# load pretrained weights# load p #
saveDir = './'




