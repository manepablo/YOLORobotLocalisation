# -*- coding: utf-8 -*-
"""
Created on Mon Oct 14 16:09:37 2019

@author: Paul
"""
def metric(y_true, y_pred):
    #This function return the mean error as well as the sqr error
    sqrerror = (abs(y_pred - y_true)**2).mean()
    meanerror = abs(y_pred - y_true).mean()
    return [sqrerror, meanerror]