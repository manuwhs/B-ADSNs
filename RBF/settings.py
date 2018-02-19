# -*- coding: utf-8 -*-
"""
Created on Tue Mar 24 01:31:58 2015

@author: montoya
"""

import numpy as np

from math_func import * # This way we import the functions directly
import paramClasses as paC

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
            self.Ytrain  = paC.get_labels(Ytrain)
            self.nO, nTr2 = self.Ytrain.shape
    
def set_Val (self, Xval, Yval):
    self.Xval = Xval
    self.Yval = Yval
    
def set_nC (self, nC):
    self.nC = nC

def set_activation_func(self, fb = "tanh", fo = "tanh"):
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
        
    if (fb == "guassian"):
        self.fb = gausskd
        self.dfb = dgausskd
        
        
def set_errFunc(self, errFunc = "square_loss"):
    self.errFunc = errFunc

def set_trainigAlg(self,  trainingAlg):
    self.trainingAlg = trainingAlg

def set_initCenters(self,  initCenters):
    self.initCenters = initCenters
    
def set_regularization(self,  regularization):
    self.regularization = regularization