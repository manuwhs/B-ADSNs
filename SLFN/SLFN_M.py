# -*- coding: utf-8 -*-
"""
Created on Tue Mar 24 01:31:58 2015

@author: montoya
"""

import numpy as np
import matplotlib.pyplot as plt
from math_func import * # This way we import the functions directly

import copy as copy

class CSLFN_M:
    # Deep Stacked Network, based on SLFN base learner that is stacked.
    def __init__(self, baseSLFN = [], mode = "single"):
        # If mode is single, only one SLFN, if it is not, one versus all.
        
        self.mode = mode
        self.set_SLFN (baseSLFN)
        self.classSLFNs = [];             # List of the SLFNs , one per class
        
    def set_SLFN (self, baseSLFN):
        # This functions sets the Base Layer, a SLFN network
        # This classifier must have all its personal parameters set and the methods:
        # classifier.fit(Xtrain, Ytrain, W)
        # classifier.soft_out(X)
        self.baseSLFN = baseSLFN
        self.classSLFNs = [];             # List of the SLFNs , one per class
        
    def set_Train (self, Xtrain, Ytrain):
        # Xtrain = M (N_samples, Ndim)
        # Ytrain = M (N_samples, Noutput)
        # Ytrain is expected to be in the form  Ytrain_i = [-1 -1 ··· 1 ··· -1 -1] for multiclass
        # This function MUST be called, ALWAYS
    
        self.Xtrain = Xtrain
        self.Ytrain = Ytrain
        
        shape = Xtrain.shape 
        self.nI = shape[1]      # Number of input dimensions
        self.nTrSa = shape[0]   # Number of training samples
        
        shape = Ytrain.shape 
        self.nO = shape[1]      # Number of output neurons 
        
    def set_Val (self, Xval, Yval):
        self.Xval = Xval
        self.Yval = Yval
        
    def train (self):
        nClasses = self.nO

        Xtrain = self.Xtrain
        Ytrain = self.Ytrain
        
        Xval = self.Xval
        Yval = self.Yval
        
        Nsamples, Ndim = Xtrain.shape;
        NsamplesVal, Ndim = Xval.shape;


        for c in range (nClasses):       # We train one SLFN per class
            # Train weak learner
            print 'Class: '+ str(c) + "/"+str(nClasses)
        
            classSLFN = copy.deepcopy(self.baseSLFN)  # Copy base layer
            
            classSLFN.set_Train(Xtrain, Ytrain[:,c].reshape(Nsamples,1))         # Set the training data
            classSLFN.set_Val(Xval, Yval[:,c].reshape(NsamplesVal,1))         # Set the training data
            
            Balance = self.get_balance(Ytrain[:,c], mode = -1)
            classSLFN.set_nH(30 + (1 - Balance) * 500)
            classSLFN.init_Weights()                    # Init weights randomly
            classSLFN.train();                          # train layer
            
            # Add the new learner to the structure
            self.classSLFNs.append(classSLFN)
            print classSLFN.score(Xtrain, Ytrain[:,c].reshape(Nsamples,1))
            print classSLFN.score(Xval, Yval[:,c].reshape(NsamplesVal,1))
            
    #########################################
    """ OUTPUT FUNCTION OBTAINING """
    #########################################
    
    def get_O (self, X):

        Nsamples, Ndim = X.shape;
        output = np.zeros((Nsamples,self.nO))

        for c in range (self.nO):
            output[:,c] = self.classSLFNs[c].get_O(X).flatten();
        
#        print output
        return output
        
    def predict_proba(self,X):
        O = self.get_O(X)    # Get the output of the net
        return O
            
    def predict(self,X):
        O = self.predict_proba(X)
                          # Multiclass classification
        
        # This will output the class index with the highest output value.
        # it works for the sigmoid, tanh and linear output.
        
        predicted = np.argmax(O, axis = 1 ) # Obtain for every sample, the class with the highest probability
#        print predicted
        return predicted;
        
    def score(self,X,Y):
        
        predicted = self.predict(X)
        
        
        N_samples,nO = Y.shape
        score = 0.0;
        
        for i in range (N_samples):
#            print predicted[i], Y[i]
            if (Y[i,predicted[i]] == 1):
                score += 1;
        
        score /= N_samples;
        return score;
        
    def get_balance (self,Y, mode = 0):
        # Counts the balance of the binary classification
    
        N = np.alen(Y)
        if (mode == 0):  # If it is 0 1
            NC1 = np.sum(Y)
            
        if (mode == -1):
            NC1 = np.sum(Y)
            NC1 = (N + NC1)/2
        
        Balance = NC1/N
        return Balance
        
        
        