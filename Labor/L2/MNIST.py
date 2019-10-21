# -*- coding: utf-8 -*-
"""
Created on Sun Oct 20 11:27:05 2019

@author: Kevin
"""

from keras.datasets import mnist
from matplotlib import pyplot as plt
import matplotlib.cm as cm
import numpy as np
import random

(x_train, y_train), (x_test, y_test) = mnist.load_data() # 2)

# 3)
for i in range(0,10):
    plt.figure("Figure " + str(i+1))
    plt.imshow(x_train[i,:,:], cmap= 'gray_r') 

# 4) 
plt.figure('Training data distribution')
fig = plt.hist(y_train, bins = np.arange(11) - 0.5, histtype='bar' ,rwidth = 0.5)
plt.xticks(range(10))
plt.figure('Test data distribution')
plt.hist(y_test, bins = np.arange(11) - 0.5, histtype='bar' ,rwidth = 0.5)
plt.xticks(range(10))
# 5)
average_image = x_train.mean(0)
plt.figure('Mean Picture')
plt.imshow(average_image, cmap= 'gray_r') 
# 6)
for i in range(0,10):
    
    img = x_train[y_train == i][random.randint(0,len(x_train[y_train == i])-1),:,:]
    img_average = img -average_image
        
    # get some image
    plt.figure(i)
    plt.imshow(img_average, cmap= 'gray_r') 

    plt.figure(i+100)
    plt.hist(img_average.flatten())
    plt.figure(i+200) 
    plt.hist(img.flatten())
    
from keras.models import Sequential
from keras.layers import Dense, Activation
from keras.utils import to_categorical
import keras.backend as K
import gc

#7)
# bitte hier mal nachschauen wieso die subtraction nicht funktioniert...
# soll diese elementweise sein also der glache vector von 'flat_img_average' soll 
# von jeder zeile aus der matrix 'flat_input_train' abgezogen werden ? 
# leider stimmt das ergebnis nicht was in 'flat_input_train_minus_averaged'steht
flat_img_average = img_average.flatten()

flat_input_train = np.reshape(x_train,(len(x_train),-1))
flat_input_train_minus_averaged = np.subtract(flat_input_train,flat_img_average)

flat_input_test = np.reshape(x_test,(len(x_test),-1))
flat_input_test_minus_averaged = np.subtract(flat_input_test,flat_img_average)

#8)
def create_model():
    
    model = Sequential()
    
    model.add(Dense(32, input_dim=flat_input_train.shape[1]))
    model.add(Dense(64,activation="relu"))
    
    # Final layer - choose the amount of classes
    model.add(Dense(10,activation="softmax"))
    return model

# 9)
y_train_cat = ...
y_test_cat = ...

# 10)
optimizers_to_test = ["rmsprop"]
for optimizer in optimizers_to_test:
    model = create_model()
    
    model.compile(optimizer=optimizer,
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])
    
    hist = model.fit(flat_input_train,y_train_cat,validation_data=\
              (flat_input_test,y_test_cat),epochs=15)
    
    plt.figure(999)
    plt.plot(hist.history["loss"])
    plt.title("Training Loss")
    
    plt.figure(998)
    plt.plot(hist.history["val_loss"])
    plt.title("Validation Loss")
    
    plt.figure(888)
    plt.plot(hist.history["acc"])
    plt.title("Trainings Accuracy")
    
    plt.figure(887)
    plt.plot(hist.history["val_acc"])
    plt.title("Validation Accuracy")

    del hist
    del model
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

# 11)
for optimizer in optimizers_to_test:
    model = create_model()
    
    model.compile(optimizer=optimizer,
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])
    
    hist = ....
    
    plt.figure(999)
    plt.plot(hist.history["loss"])
    plt.title("Training Loss")
    
    plt.figure(998)
    plt.plot(hist.history["val_loss"])
    plt.title("Validation Loss")
    
    plt.figure(888)
    plt.plot(hist.history["acc"])
    plt.title("Trainings Accuracy")
    
    plt.figure(887)
    plt.plot(hist.history["val_acc"])
    plt.title("Validation Accuracy")

    del hist
    del model
    K.clear_session()
    gc.collect()

new_legend = optimizers_to_test + [i + " mean" for i in optimizers_to_test]
plt.figure(999)
plt.legend(new_legend)
plt.figure(998)
plt.legend(new_legend)
plt.figure(888)
plt.legend(new_legend)
plt.figure(887)
plt.legend(new_legend)
