# -*- coding: utf-8 -*-
"""
Created on Mon Oct 14 16:17:54 2019

@author: Paul
"""

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
import matplotlib.pyplot as plt

# 1) Try to understand the differences between training and testing dataset
(x_train, y_train), (x_test, y_test) = boston_housing.load_data()

y_train_class = [int(i) for i in y_train]


# 2) Look at a histogram of the data

result_errors = []
n_md = 30
for i in range(1,n_md):
    print(i)
    clf = RandomForestClassifier(n_estimators=100, max_depth=i,
                                  random_state=0)
    
    clf.fit(x_train,y_train_class)
    result = clf.predict(x_test)
    [sqrerror, meanerror] = metric(y_test, result)
    result_errors.append(meanerror)

plt.plot(list(np.arange(1,n_md,1)),result_errors)