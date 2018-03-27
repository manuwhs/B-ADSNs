# -*- coding: utf-8 -*-
"""
Created on Tue Mar 24 01:31:58 2015

@author: montoya
"""

import numpy as np


#########################################################
""" Generic NON-class oriented functions """
#########################################################
def get_Z (X,W,b):
    # Obtains the activation value of the hidden layer
    # Z = nSa x nH  Output of every Hidden neuron to every Sample
#    print X.shape
#    print W.shape
    nSa, nDim = X.shape   # Parameter of the input matrix
#    print X.shape, W.shape
    Z = np.dot(X, W);   # Get the activation of the hidden layer
    Z += np.tile (b, (nSa,1))
    
    return Z


#########################################
""" PROPAGATIONS OUTPUT """
#########################################

def get_Zh (self, X):
    Z = get_Z (X, self.Wh, self.bh)
#    print Z.shape
    return Z
    
def get_H (self, X):
    # H = nSa x nH  Output of every Hidden neuron to every Sample
    Z = self.get_Zh(X)
    H = self.fh(Z)       # Apply the transfer function
#    print H
    self.H = H          # Output Hidden Matrix
    return H

#########################################
""" OUTPUT FUNCTION OBTAINING """
#########################################
def get_Zo (self, X):
    H = self.get_H(X)
    Zo = get_Z (H, self.Wo, self.bo)
#        print Zo
    return Zo
    
def get_O (self, X):
    Zo = self.get_Zo(X)
    O = self.fo(Zo)
    self.O = O
#        print O
    return O

#########################################
""" EARLY STOPPING """
#########################################

def evaluate_stop(self, e, error_tr, error_val):
    ws = 5;   # Window size
    percent = 0.05
    if (e > 2*ws):
        error_prev = np.sum(error_val[e-ws:e]) / ws
        error_new = np.sum(error_val[e-ws*2:e-ws]) / ws
        
        if (error_new - error_prev < -percent): # If there has been an empoverishment of percent
#            print error_prev, error_new
            return 1;
    
    return 0
    

