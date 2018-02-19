# -*- coding: utf-8 -*-
"""
Created on Tue Mar 24 01:31:58 2015

@author: montoya
"""

import numpy as np

def get_Z (X,W,b):
    # Obtains the activation value of the hidden layer
    # Z = nSa x nH  Output of every Hidden neuron to every Sample
#    print X.shape
#    print W.shape
    nSa, nDim = X.shape   # Parameter of the input matrix
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
    
def get_G (self, X):
    # H = nSa x nG  Output of every Hidden neuron to every Sample
    # Calculate output of the hidden nuerons
    # Activations of RBFs
    G = np.zeros((X.shape[0], self.nC), float)
    for ci, c in enumerate(self.centers):
        for xi, x in enumerate(X):
            G[xi,ci] = self.fb(c, x, self.beta)
    return G

#########################################
""" OUTPUT FUNCTION OBTAINING """
#########################################
def get_Zo (self, X):
    H = self.get_G(X)
    Zo = get_Z (H, self.Wo, self.bo)
#        print Zo
    return Zo
    
def get_O (self, X):
    Zo = self.get_Zo(X)
    O = self.fo(Zo)
    self.O = O
#        print O
    return O

