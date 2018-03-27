# -*- coding: utf-8 -*-
"""
Created on Tue Mar 24 01:31:58 2015

@author: montoya
"""

import numpy as np

""" SAME AS THE SLFN """

###########################################################################################################################
######################################### sklearn INTEFACE !!  ##########################################################
###########################################################################################################################

def fit (self, Xtrain, Ytrain):
    self.set_Train (Xtrain, Ytrain)
    self.train()

def predict_proba(self,X):
    O = self.get_O(X)    # Get the output of the net
    return O
    
def soft_out(self,X):
    O = self.get_O(X)    # Get the output of the net
    return O
    
def predict(self,X):
    O = self.predict_proba(X)
    if (self.nO == 1):      # Standard binary classification
    
        if (self.outMode == -1):  # tanh goes from -1 to 1
            predicted = np.sign(O)
            
        if (self.outMode == 0):  # sigmoid goes from 0 to 1
            predicted = np.zeros((np.alen(O),1)) 
            for i in range (np.alen(O)):
                if (O[i] > 0.5):
                    predicted[i] = 1

    else:                   # Multiclass classification
        predicted = np.argmax(O, axis = 1 ) # Obtain for every sample, the class with the highest probability
#        print predicted
    return predicted;
    
def score(self,X,Y):
    
    predicted = self.predict(X)
    N_samples,nO = Y.shape
    score = 0.0;
#        print N_samples
#        print X.shape
#        
    if (self.nO == 1):
        for i in range (N_samples):
#            print predicted[i], Y[i]
            if (Y[i] == predicted[i]):
                score += 1;
    else: # Multiclass case
        for i in range (N_samples):
#            print predicted[i], Y[i]
            if (Y[i,predicted[i]] == 1):
                score += 1;
                
    score /= N_samples;
    return score;
