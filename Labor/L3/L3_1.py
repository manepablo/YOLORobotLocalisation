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
import random

(x_train, y_train), (x_test, y_test) = mnist.load_data() # 2)

# 3)
for i in range(0,10):
    plt.figure("Figure " + str(i+1))
    plt.title("Example: " + str(i+1))
    plt.imshow(x_train[i,:,:], cmap= 'gray_r') 

# 4) 

plt.figure('Training data distribution')
plt.title('Training data distribution')
fig = plt.hist(y_train, bins = np.arange(11) - 0.5, histtype='bar' ,rwidth = 0.5)
plt.xticks(range(10))


plt.figure('Test data distribution')
plt.title('Test data distribution')
plt.hist(y_test, bins = np.arange(11) - 0.5, histtype='bar' ,rwidth = 0.5)
plt.xticks(range(10))
# 5)

average_image = x_train.mean(0)
plt.figure('Mean Picture')
plt.title('Average Image')
plt.imshow(average_image, cmap= 'gray_r') 

# 6)
for i in range(0,10):
    
    img = x_train[y_train == i][random.randint(0,len(x_train[y_train == i])-1),:,:]
    img_average = img - average_image
        
    # get some image
    plt.figure(i)
    plt.title('Class ' + str(i) + ' - Average Image')
    plt.imshow(img_average, cmap= 'gray_r') 

    plt.figure(i+100)
    plt.title('Hist of class ' + str(i))
    plt.hist(img.flatten())
    
    plt.figure(i+200)
    plt.title('Hist of class ' + str(i) + ' - average img')
    plt.hist(img_average.flatten())
    
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
flat_input_train_minus_avrg = flat_input_train - flat_img_average


flat_input_test = np.reshape(x_test,(len(x_test),-1))
flat_input_test_minus_avrg = flat_input_test - flat_img_average

#8)
def create_model(m_shape = 784):
    
    inputs = Input(shape=(m_shape,))
    
    # a layer instance is callable on a tensor, and returns a tensor
    output_1 = Dense(32, activation='relu', )(inputs)
    output_2 = Dense(64, activation='relu')(output_1)
    predictions = Dense(10, activation='softmax')(output_2)
    
    # This creates a model that includes
    # the Input layer and three Dense layers
    model = Model(inputs=inputs, outputs=predictions)
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
    
    hist = model.fit(flat_input_train,y_train_cat,validation_data=\
              (flat_input_test,y_test_cat),epochs=10)
    
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

## 11)
# durchführung mit flat_input_train_minus_avrg
optimizers_to_test = ['adam']
for optimizer in optimizers_to_test:
    print(optimizer)
    model = create_model(m_shape = flat_input_train.shape[1])
    
    model.compile(optimizer=optimizer,
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])
    
    hist = model.fit(flat_input_train_minus_avrg,y_train_cat,validation_data=\
              (flat_input_test_minus_avrg,y_test_cat),epochs=10)
    
    plt.figure(999)
    plt.plot(hist.history["loss"])
    plt.title("Training Loss " )
    
    plt.figure(998)
    plt.plot(hist.history["val_loss"])
    plt.title("Validation Loss " )
    
    plt.figure(888)
    plt.plot(hist.history["accuracy"])
    plt.title("Trainings Accuracy  ")
    
    plt.figure(887)
    plt.plot(hist.history["val_accuracy"])
    plt.title("Validation Accuracy ")

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





