# -*- coding: utf-8 -*-
"""
Created on Tue Mar 24 01:31:58 2015

@author: montoya
"""

import numpy as np


#########################################
""" OUTPUT FUNCTION OBTAINING """
#########################################

def get_O (self, X, lEnd = -1):
    # Gets the output of the system for the first lEnd layers
    """ It only propagates the Activation of outputs """
    
    if (lEnd < - 0.5):
        lEnd = self.nL 
#        print lEnd
    
    Nsamples, Ndim = X.shape;
    output = np.zeros((Nsamples,1))
    
    PrevZo = []
    for l in range (lEnd):
        if (l == 0):
            output = self.layers[l].get_Zo(X);
        else:
            output = self.layers[l].get_Zo(X, self.layers[l].fg(PrevZo));
            
        PrevZo = output  # Propagation
    
    # We apply the last output right
    output = self.layers[-1].fo(output)
    
#        print output
    return output
        