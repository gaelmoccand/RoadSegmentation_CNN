# -*- coding: utf-8 -*-
"""a function of ploting figures."""
import numpy as np
import matplotlib.pyplot as plt
from proj1_helpers import *
from implementations import *
import numpy.ma as ma

def submission(train_data_path,test_data_path):
    """Creates a sample-submission.csv file using the data set defined by data_path"""
    y_data,x_data,ind=load_csv_data(train_data_path) #load the data set
    tx,y=preprocessing(x_data,y_data)
    polyN=6
    tx_tr = build_poly(tx, polyN) # use polynomial basis
    w,loss = ridge_regression(y, tx_tr, 0.0001) #use the ridge regression to compute the w  
    
    # Now using the test set
    y_data_test,x_data_test,ind=load_csv_data(test_data_path)
    tx_tst,y_tst = preprocessing(x_data_test,y_data_test)
    tx_tst_p = build_poly(tx_tst, polyN) # use polynomial basis
    y_prime = predict_labels(w,tx_tst_p)#use the w found with the ridge regression
    create_csv_submission(ind, y_prime, 'sample-submission.csv')


def preprocessing(x_data,y_data):
    """Do the preprossing by santardazing and taking care of -999 values and using a log of inverse transform"""
    feat_log_inv = np.log(1 / (1 + x_data[:, [0,1,2,3,4,5,7,8,9,10,12,13,16,19,21,23,26]]))
    x_data[:,[0,1,2,3,4,5,7,8,9,10,12,13,16,19,21,23,26]]=feat_log_inv
    y_samp =y_data
    x_samp = x_data
    x_samp, mean_x, std_x=standardizeNan(x_samp)
    tx = np.c_[np.ones((y_samp.shape[0], 1)), x_samp]
    y = np.expand_dims(y_samp, axis=1)
    return tx,y
    
def standardizeNan(x):
    """Standardize the original data set."""
    x[np.where(x == -999)] = np.nan
    x=np.where(np.isnan(x),ma.array(x,mask=np.isnan(x)).mean(axis=0),x)
    mean_x = np.nanmean(x,axis=0)
    x = x - mean_x
    std_x = np.nanstd(x,axis=0)
    x = x / std_x
    return x, mean_x, std_x

def build_poly(x, degs):
    """polynomial basis functions."""
    poly = np.ones((len(x), 1))
    for deg in range(1, degs+1):
        poly = np.c_[poly, x**deg]
    return poly


