# -*- coding: utf-8 -*-
"""
Created on Mon Oct 28 15:30:59 2019

@author: Paul
"""

from keras.layers import Input, Dense
from keras.models import Model

# -*- coding: utf-8 -*-
"""
Created on Sun Oct 20 11:27:05 2019

@author: Kevin
"""

from keras.datasets import mnist
from matplotlib import pyplot as plt
import matplotlib.cm as cm
import numpy as np


(x_train, y_train), (x_test, y_test) = mnist.load_data() # 2)
average_image = x_train.mean(0)

x_train = x_train.reshape(x_train.shape[0], x_train.shape[1], x_train.shape[2],1)
x_test = x_test.reshape(x_test.shape[0], x_test.shape[1], x_test.shape[2],1)

from keras.models import Sequential
from keras.layers import Dense, Activation, Concatenate, Dropout, Conv2D, MaxPool2D, Flatten
from keras.utils import to_categorical
import keras.backend as K
import gc

#8)
def create_model(m_shape = (28, 28, 1)):
    
    model = Sequential()
    model.add(Conv2D(filters = 32, input_shape = m_shape, kernel_size = (5,5), activation = 'relu', padding = 'same'))
    model.add(Conv2D(filters = 32, kernel_size = (5,5), activation = 'relu', padding = 'same'))
    model.add(Conv2D(filters = 32, kernel_size = (5,5), activation = 'relu', padding = 'same'))
    model.add(Conv2D(filters = 32, kernel_size = (5,5), activation = 'relu', padding = 'same'))  
    model.add(MaxPool2D(pool_size=(2,2)))
    model.add(Dropout(0.25))
    model.add(Conv2D(filters = 32, kernel_size = (5,5), activation = 'relu', padding = 'same'))
    model.add(Conv2D(filters = 32, kernel_size = (5,5), activation = 'relu', padding = 'same'))
    model.add(Conv2D(filters = 32, kernel_size = (5,5), activation = 'relu', padding = 'same')) 
    model.add(Conv2D(filters = 32, kernel_size = (5,5), activation = 'relu', padding = 'same'))
    model.add(MaxPool2D(pool_size=(2,2)))        
    model.add(Flatten())
    model.add(Dense(64, activation = 'relu'))
    model.add(Dropout(0.5))
    model.add(Dense(10, activation = 'softmax'))

    return model

# 9)
y_train_cat = to_categorical(y_train)
y_test_cat = to_categorical(y_test)

# 10)
optimizers_to_test = ['adam']

for optimizer in optimizers_to_test:
    print(optimizer)
    model = create_model()
    
    model.compile(optimizer=optimizer,
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])
    
    hist = model.fit(x_train,y_train_cat,validation_data=\
              (x_test,y_test_cat),epochs=12)
    
    plt.figure(999)
    plt.plot(hist.history["loss"])
    plt.title("Training Loss - " + optimizer)
    
    plt.figure(998)
    plt.plot(hist.history["val_loss"])
    plt.title("Validation Loss - " + optimizer)
    
    plt.figure(888)
    plt.plot(hist.history["accuracy"])
    plt.title("Trainings Accuracy - " + optimizer)
    
    plt.figure(887)
    plt.plot(hist.history["val_accuracy"])
    plt.title("Validation Accuracy - " + optimizer)

    del hist
    K.clear_session()
    gc.collect()


plt.figure(999)
plt.legend(optimizers_to_test)
plt.figure(998)
plt.legend(optimizers_to_test)
plt.figure(888)
plt.legend(optimizers_to_test)
plt.figure(887)
plt.legend(optimizers_to_test)






