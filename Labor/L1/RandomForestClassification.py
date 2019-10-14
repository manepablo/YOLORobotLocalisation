# -*- coding: utf-8 -*-
"""
Created on Mon Oct 14 11:27:47 2019

@author: khoefle
"""
from utils import metric
import numpy as np
# See: https://keras.io/datasets/#boston-housing-price-regression-dataset
from keras.datasets import boston_housing
from sklearn.ensemble import RandomForestClassifier

# 1) Try to understand the differences between training and testing dataset
(x_train, y_train), (x_test, y_test) = boston_housing.load_data()

y_train_class = [int(i) for i in y_train]


# 2) Look at a histogram of the data

results = []

# Try to understand what the criterion means, why is it mse? 
# https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.RandomForestClassifier.html
clf = RandomForestClassifier(n_estimators=100, max_depth=2,
                              random_state=0)

clf.fit(x_train,y_train_class)
result = clf.predict(x_test)


# Write a function to describe the error
[sqrerror, meanerror] = metric(y_test, result)

print('\nSQR ERROR  :  ' + str(sqrerror))
print('\nMEAN ERROR :  ' + str(meanerror))




