# -*- coding: utf-8 -*-
"""
Created on Mon Dec  9 15:18:46 2019

@author: Paul
"""
from tensorflow import keras

from tensorflow.keras import backend as K
import tensorflow as tf
import time
#from keras.utils import plot_model


#2D Class
class ObjectLocalizer2D ( object ) :

	def __init__(self, input_shape):

		#tf.logging.set_verbosity(tf.logging.ERROR)
		alpha = 0.2

		def calculate_iou(target_boxes, pred_boxes):
			xA = K.maximum(target_boxes[..., 0], pred_boxes[..., 0])
			yA = K.maximum(target_boxes[..., 1], pred_boxes[..., 1])
			xB = K.minimum(target_boxes[..., 2], pred_boxes[..., 2])
			yB = K.minimum(target_boxes[..., 3], pred_boxes[..., 3])
			interArea = K.maximum(0.0, xB - xA) * K.maximum(0.0, yB - yA)
			boxAArea = (target_boxes[..., 2] - target_boxes[..., 0]) * (target_boxes[..., 3] - target_boxes[..., 1])
			boxBArea = (pred_boxes[..., 2] - pred_boxes[..., 0]) * (pred_boxes[..., 3] - pred_boxes[..., 1])
			iou = interArea / (boxAArea + boxBArea - interArea)
			return iou

		def custom_loss(y_true, y_pred):
			mse = tf.losses.mean_squared_error(y_true, y_pred)
			iou = calculate_iou(y_true, y_pred)
			#return mse + (1 - iou)
			return (1 - iou)

		def iou_metric(y_true, y_pred):
			return calculate_iou(y_true, y_pred)

		model_layers = [
			keras.layers.Conv2D(16, kernel_size=(3, 3), strides=1, input_shape=input_shape),
			keras.layers.LeakyReLU(alpha=alpha),
			keras.layers.Conv2D(16, kernel_size=(3, 3), strides=1),
			keras.layers.LeakyReLU(alpha=alpha),
			keras.layers.MaxPooling2D(pool_size=(2, 2)),

			keras.layers.Conv2D(32, kernel_size=(3, 3), strides=1),
			keras.layers.LeakyReLU(alpha=alpha),
			keras.layers.Conv2D(32, kernel_size=(3, 3), strides=1),
			keras.layers.LeakyReLU(alpha=alpha),
			keras.layers.MaxPooling2D(pool_size=(2, 2)),

			keras.layers.Conv2D(64, kernel_size=(3, 3), strides=1),
			keras.layers.LeakyReLU(alpha=alpha),
			keras.layers.Conv2D(64, kernel_size=(3, 3), strides=1),
			keras.layers.LeakyReLU(alpha=alpha),
			keras.layers.MaxPooling2D(pool_size=(2, 2)),

			keras.layers.Conv2D(128, kernel_size=(3, 3), strides=1),
			keras.layers.LeakyReLU(alpha=alpha),
			keras.layers.Conv2D(128, kernel_size=(3, 3), strides=1),
			keras.layers.LeakyReLU(alpha=alpha),
			keras.layers.MaxPooling2D(pool_size=(2, 2)),

			keras.layers.Conv2D(256, kernel_size=(3, 3), strides=1),
			keras.layers.LeakyReLU(alpha=alpha),
			keras.layers.Conv2D(256, kernel_size=(3, 3), strides=1),
			keras.layers.LeakyReLU(alpha=alpha),
			keras.layers.MaxPooling2D(pool_size=(2, 2)),

			keras.layers.Flatten(),

			keras.layers.Dense(1240),
			keras.layers.LeakyReLU(alpha=alpha),
			keras.layers.Dense(640),
			keras.layers.LeakyReLU(alpha=alpha),
			keras.layers.Dense(480),
			keras.layers.LeakyReLU(alpha=alpha),
			keras.layers.Dense(120),
			keras.layers.LeakyReLU(alpha=alpha),
			keras.layers.Dense(62),
			keras.layers.LeakyReLU(alpha=alpha),

			keras.layers.Dense(5),
			keras.layers.LeakyReLU(alpha=alpha),
		]

		self.__model = keras.Sequential(model_layers)
		self.__model.compile(
			optimizer=keras.optimizers.Adam(lr=0.0001),
			loss=custom_loss,
			metrics=[iou_metric]
		) 
		#plot_model(self.__model, to_file='model.png')
    



	def fit(self, X, Y, hyperparameters):
		print(Y)           
		initial_time = time.time()
		self.__model.fit(X, Y,
						 batch_size=hyperparameters['batch_size'],
						 epochs=hyperparameters['epochs'],
						 callbacks=hyperparameters['callbacks'],
						 validation_data=hyperparameters['val_data']
						 )
		final_time = time.time()
		eta = (final_time - initial_time)
		time_unit = 'seconds'
		if eta >= 60:
			eta = eta / 60
			time_unit = 'minutes'
		self.__model.summary()
		print('Elapsed time acquired for {} epoch(s) -> {} {}'.format(hyperparameters['epochs'], eta, time_unit))


	def evaluate(self, test_X, test_Y):
		return self.__model.evaluate(test_X, test_Y)


	def predict(self, X):
		predictions = self.__model.predict(X)
		return predictions


	def save_model(self, file_path):
		self.__model.save(file_path)


	def load_model(self, file_path):
		self.__model = keras.models.load_model(file_path)

	def load_model_weights(self , file_path ) :
		self.__model.load_weights( file_path )
        
#3D Class
import math
class ObjectLocalizer3D ( object ) :

	def __init__(self, input_shape):
		alpha = 0.2
        
		def custom_loss(y_true, y_pred):
			mse = tf.losses.mean_squared_error(y_true, y_pred)			
			return mse 

		def calculate_iou(target_boxes, pred_boxes):
			
			xA = (target_boxes[..., 0] + target_boxes[..., 2] + target_boxes[..., 4] + target_boxes[..., 6] + target_boxes[..., 8] + target_boxes[..., 10] + target_boxes[..., 12] + target_boxes[..., 14])/8
			yA = (target_boxes[..., 1] + target_boxes[..., 3] + target_boxes[..., 5] + target_boxes[..., 7] + target_boxes[..., 9] + target_boxes[..., 11] + target_boxes[..., 13] + target_boxes[..., 15])/8      
			xB = (pred_boxes[..., 0] + pred_boxes[..., 2] + pred_boxes[..., 4] + pred_boxes[..., 6] + pred_boxes[..., 8] + pred_boxes[..., 10] + pred_boxes[..., 12] + pred_boxes[..., 14])/8
			yB = (pred_boxes[..., 1] + pred_boxes[..., 3] + pred_boxes[..., 5] + pred_boxes[..., 7] + pred_boxes[..., 9] + pred_boxes[..., 11] + pred_boxes[..., 13] + pred_boxes[..., 15])/8
			ret = (((xA-xB)*(xA-xB)+(yA-yB)*(yA-yB)))
			print(ret)
			return ret
       
        
		def iou_metric(y_true, y_pred):
			return calculate_iou(y_true, y_pred)
        
		print("Intialize Network")

        
		model_layers = [
			keras.layers.Conv2D(16, kernel_size=(3, 3), strides=1, input_shape=input_shape),
			keras.layers.LeakyReLU(alpha=alpha),
			keras.layers.Conv2D(16, kernel_size=(3, 3), strides=1),
			keras.layers.LeakyReLU(alpha=alpha),
			keras.layers.MaxPooling2D(pool_size=(2, 2)),

			keras.layers.Conv2D(32, kernel_size=(3, 3), strides=1),
			keras.layers.LeakyReLU(alpha=alpha),
			keras.layers.Conv2D(32, kernel_size=(3, 3), strides=1),
			keras.layers.LeakyReLU(alpha=alpha),
			keras.layers.MaxPooling2D(pool_size=(2, 2)),

			keras.layers.Conv2D(64, kernel_size=(3, 3), strides=1),
			keras.layers.LeakyReLU(alpha=alpha),
			keras.layers.Conv2D(64, kernel_size=(3, 3), strides=1),
			keras.layers.LeakyReLU(alpha=alpha),
			keras.layers.MaxPooling2D(pool_size=(2, 2)),

			keras.layers.Conv2D(128, kernel_size=(3, 3), strides=1),
			keras.layers.LeakyReLU(alpha=alpha),
			keras.layers.Conv2D(128, kernel_size=(3, 3), strides=1),
			keras.layers.LeakyReLU(alpha=alpha),
			keras.layers.MaxPooling2D(pool_size=(2, 2)),

			keras.layers.Conv2D(256, kernel_size=(3, 3), strides=1),
			keras.layers.LeakyReLU(alpha=alpha),
			keras.layers.Conv2D(256, kernel_size=(3, 3), strides=1),
			keras.layers.LeakyReLU(alpha=alpha),
			keras.layers.MaxPooling2D(pool_size=(2, 2)),

			keras.layers.Flatten(),

			keras.layers.Dense(1240),
			keras.layers.LeakyReLU(alpha=alpha),
			keras.layers.Dense(640),
			keras.layers.LeakyReLU(alpha=alpha),
			keras.layers.Dense(480),
			keras.layers.LeakyReLU(alpha=alpha),
			keras.layers.Dense(120),
			keras.layers.LeakyReLU(alpha=alpha),
			keras.layers.Dense(62),
			keras.layers.LeakyReLU(alpha=alpha),

			keras.layers.Dense( 17 ),
			keras.layers.LeakyReLU(alpha=alpha),
		]
        
		print("Create Network")
		self.__model = keras.Sequential(model_layers)
		print("Compile Network")
		self.__model.compile(
			optimizer=keras.optimizers.Adam(lr=0.0001),
			loss=custom_loss , 
			metrics=[iou_metric]
		)

	def fit(self, X, Y, hyperparameters):
		initial_time = time.time()
		print(Y)        
		self.__model.fit(X, Y,
						 batch_size=hyperparameters['batch_size'],
						 epochs=hyperparameters['epochs'],
						 callbacks=hyperparameters['callbacks'],
						 validation_data=hyperparameters['val_data']
						 )
		final_time = time.time()
		eta = (final_time - initial_time)
		time_unit = 'seconds'
		if eta >= 60:
			eta = eta / 60
			time_unit = 'minutes'
		self.__model.summary()
		print('Elapsed time acquired for {} epoch(s) -> {} {}'.format(hyperparameters['epochs'], eta, time_unit))


	def evaluate(self, test_X, test_Y):
		return self.__model.evaluate(test_X, test_Y)


	def predict(self, X):
		predictions = self.__model.predict(X)
		return predictions


	def save_model(self, file_path):
		self.__model.save(file_path)


	def load_model(self, file_path):
		self.__model = keras.models.load_model(file_path)

	def load_model_weights(self , file_path ) :
		self.__model.load_weights( file_path )        