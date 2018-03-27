# -*- coding: utf-8 -*-
"""
Created on Tue Mar 24 01:31:58 2015

@author: montoya
"""

import numpy as np
from math_func import *

####################################
""" INPUT PARAMETER FUNCTIONS """
###################################

def set_Train (self, Xtrain, Ytrain):
    # Xtrain = M (N_samples, Ndim)
    # Ytrain = M (N_samples, Noutput)
    # Ytrain is expected to be in the form  Ytrain_i = [-1 -1 ··· 1 ··· -1 -1] for multiclass
    # This function MUST be called, ALWAYS

    self.Xtrain = Xtrain
    self.Ytrain = Ytrain
    
    # Number of input dimensions
    # Number of training samples
    self.nTrSa,self.nI  = Xtrain.shape 

    # Number of output neurons 
    self.nO = Ytrain.size / Ytrain.shape[0]
    
#    if (self.nTrSa != nTr2):
#        print "Number of samples and labels does not match"

    # CHECK IF IT IS MULTICLASS AND WELL GIVEN
    if (self.nO == 1):   # Check if the paramters were given as Y = [3 4 1 5 2 4 3]
        if (np.alen(np.unique(Ytrain)) > 2): # If there are more than 2 clases
            self.Ytrain  = self.get_labels(Ytrain)
            self.nO, nTr2 = self.Ytrain.shape
    
def set_Val (self, Xval, Yval):
    self.Xval = Xval
    self.Yval = Yval

def set_Test (self, Xtest, Ytest):
    # For obtaining the score of an specific test data
    self.Xtest = Xtest
    self.Ytest = Ytest
    
def set_nH (self, nH):
    self.nH = nH

def set_activation_func(self, fh = "tanh", fo = "tanh"):
    if (fo == "sigmoid"):
        self.fo = sigm
        self.dfo = dsigm
        self.outMode = 0   # Mode of the negative class
        
    if (fo == "tanh"):
        self.fo = tanh
        self.dfo = dtanh
        self.outMode = -1   # Mode of the negative class

    if (fo == "linear"):
        self.fo = linear
        self.dfo = dlinear
        self.outMode = -1   # Mode of the negative class
        
    if (fh == "sigmoid"):
        self.fh = sigm
        self.dfh = dsigm
        
    if (fh == "tanh"):
        self.fh = tanh
        self.dfh = dtanh
        
    if (fh == "linear"):
        self.fh = linear
        self.dfh = dlinear
        
def set_errFunc(self, errFunc = "square_loss"):
    self.errFunc = errFunc

def set_trainigAlg(self,  trainingAlg):
    self.trainingAlg = trainingAlg

def set_initDistrib(self,  initDistrib):
    self.initDistrib = initDistrib
    
def set_regularization(self,  regularization):
    self.regularization = regularization
    
def set_stopCrit(self,  stopCrit):
    self.stopCrit = stopCrit

def set_D(self, D):  # Trainining Samples distribution
    self.D = D
    self.D_flag = 1 # Activate distributions
    