# -*- coding: utf-8 -*-
"""
Created on Tue Mar 24 01:31:58 2015

@author: montoya
"""

import numpy as np
from sklearn import cross_validation

from time import time    # For the random seed

def train_CV (self, r):  # Different because of realimentation score(X,GX)
    
    total_Xtrain = self.Xtrain
    total_GXtrain = self.GXtrain
    total_Ytrain = self.Ytrain

    ## Get the random seed and use it
    if (self.InitRandomSeed == -1): # If no seed is specified
        self.RandomSeed[r] = int((time()%1 * 100000))
        np.random.seed(self.RandomSeed[r])
    else:
        self.RandomSeed[r] = self.InitRandomSeed
        np.random.seed(self.RandomSeed[r])
        
    TrError = 0;
    ValError = 0;
    TstError = 0;
    
    if (self.CV == 1):  
        # If the validation is performed with just the training set
        # Then the validation set is the original self.Xval. 
        """ Why you may ask ?? """ 
        # In other aggregate solutions, like Boosting, the CV is done
        # over the whole structure, not layer by layer. In this cases,
        # the CV of the SLFN will be 1 always and its the Boosting "train"
        # the one in charge for changing the Validation set and training set.
        
        self.train_once()
        
        TrError += self.score(self.Xtrain,self.GXtrain, self.Ytrain)
        ValError += self.score(self.Xval,self.GXval, self.Yval)
        if (self.Xtest != []):     # If there is a test dataset.
            TstError += self.score(self.Xtest,self.GXtest,self.Ytest)
    
    if (self.CV > 1):

        stkfold = cross_validation.StratifiedKFold(total_Ytrain.ravel(), n_folds = self.CV)
        for train_index, val_index in stkfold:
#                print train_index
            self.set_Train(total_Xtrain[train_index],total_Ytrain[train_index])
            self.set_Val(total_Xtrain[val_index],total_Ytrain[val_index])
            
            self.set_GTrain(total_GXtrain[train_index])
            self.set_GVal(total_GXtrain[val_index])
            
            self.train_once()
        
            TrError += self.score(self.Xtrain,self.GXtrain, self.Ytrain)
            ValError += self.score(self.Xval,self.GXval, self.Yval)
            
            if (self.Xtest != []):     # If there is a test dataset.
                TstError += self.score(self.Xtest,self.GXtest,self.Ytest)
            
    self.TrError[r] = TrError / self.CV
    self.ValError[r] = ValError / self.CV
    self.TstError[r] = TstError / self.CV
    
    self.Xtrain = total_Xtrain   # Restore the original Xtrain
    self.GXtrain = total_GXtrain   # Restore the original Xtrain
    self.Ytrain = total_Ytrain 
    
    