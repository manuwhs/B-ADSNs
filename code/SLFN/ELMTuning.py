# -*- coding: utf-8 -*-
"""
Created on Tue Mar 24 01:31:58 2015

@author: montoya
"""
import numpy as np

def ELM_train (self, param):
    type_bias = param[0]
    
    H = self.get_H(self.Xtrain)     # Get the hidden output matrix
#    print H
    ##############  BOOSTING  ##############
    if (self.D_flag == 1):   # If we have given Weights to the samples
        W_root = np.sqrt(self.D)
        H = H * W_root
    
    if (type_bias == "no_bias"):
        self.bo = np.zeros((1,self.nO))  # In the standard ELM, these does not count
        
        Hinv = np.linalg.pinv(H)         # Get the inverse of the matrix 
        beta = np.dot(Hinv,self.Ytrain)  # Get output weights
        
        self.Wo = beta                   # Write the output weights into the structure
    
    if (type_bias == "bias"):
        
        Nsa, Nh = H.shape
        H = np.concatenate((H, np.ones((Nsa,1))),axis = 1)   # Add a bias value to every sample
        Hinv = np.linalg.pinv(H)                # Get the inverse of the matrix 
        beta = np.dot(Hinv,self.Ytrain)         # Get output weights and bias
        
        self.Wo = beta[:Nh:]                 # Obtain the output weights
        self.bo = beta[Nh,:]                 # Obtain the output bias

#        print self.bo

def ELMT_train (self, param):
    # Training using the FineTuning algorithm which is a mixture of ELM and BMBP

    n_epoch = param[0]
    step = param[1]
    partitions = param[2]
    
    bias = param[3]

    # SET THE OUTPUT LAYER
    self.ELM_train([bias])
    self.BMBP_train([n_epoch, step, partitions,"si al linear decrease"])
#    self.ELM_train([bias])