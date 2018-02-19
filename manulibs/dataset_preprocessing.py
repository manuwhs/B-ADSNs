# -*- coding: utf-8 -*-
"""
Created on Tue Mar 24 01:31:58 2015

@author: montoya
"""

# The aim of this file is to contain functions that will read particular
# datasets and preprocess them to obtian the X and Y labels.

# We use pandas library to read CSV data.
import pandas as pd
import numpy as np


#==============================================================================
# data = pandas.read_csv('../dataR/winequality-red.csv', sep = ';')


def abalone_dataset(file_dir = "./data/abalone/abalone.data"):
    
    data = pd.read_csv(file_dir, sep = ',',header = None, names = None) 
    # names = None coz the file does not contain column names
    Nsamples, Ndim = data.shape   # Get the number of bits and attr
    
    # The first column has tha classes M,F, I. We wanto to change them to 0,1,2
    
    first_col = data.ix[:,0]  # COGE LA PRIMERA PUTA COLUMNA !!!!
    for i in range(Nsamples):
        if (first_col[i]== "M"):
            first_col[i]= 0
        elif (first_col[i] == "F"):
            first_col[i] = 1  
        elif (first_col[i] == "I"):
            first_col[i] = 2
        else:
            print "No detectdao !!!!"
            
    data_np = np.array(data, dtype = float).reshape(Nsamples, Ndim)
    
    X = data_np[:,:-1]
    Y = data_np[:,-1].reshape((Nsamples,1))
    
    # Y goes from 1 to 29 or roughly, integer, number of rings.
    # To make it a classification problem we slit it into 2 sets of more or less
    # same number of samples.
    
    ring_class = np.unique(Y)
    num_classes = len(ring_class)
    
    ring_count = np.zeros(num_classes)
    for i in range (num_classes):
        
        ring_count[i] = Nsamples - np.count_nonzero(Y - ring_class[i])
    
#    print np.sum(ring_count)
    # Ring classes is a sorted list of the Nrings and ring counts contains
    # the number of instances of each class.
    
    # Now we grop them in to classes with roughly the same number of samples
    num_class_0 = 0
    for ring_indx in range (num_classes):
        num_class_0 += ring_count[ring_indx]
        if (num_class_0 > Nsamples/2):
            break;
            
#    print num_class_0
#    print Nsamples - num_class_0
    
#    print ring_indx
    
    # Now we set the labels of Y accordingly:
    
    for i in range (Nsamples):
        if (Y[i] <= ring_class[ring_indx]):
            Y[i] = -1
        else:
            Y[i] = 1
            
    return X, Y
    
#X, Y = abalone_dataset()


def obtain_train_test (X, Y, train_ratio):
    
####################################################################
#################### SPLIT TRAIN TEST data #########################
#################################################################
    order = np.arange(np.shape(X)[0],dtype=int) # Create array of index
    order = np.random.permutation(order)        # Randomize the array of index
    
    Ntrain = round(train_ratio*np.shape(X)[0])    # Number of samples used for training
    Ntest = len(order)-Ntrain                  # Number of samples used for testing

    
    return Xtrain, Xtest, Ytrain, Ytest

