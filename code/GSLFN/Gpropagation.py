# -*- coding: utf-8 -*-
"""
Created on Tue Mar 24 01:31:58 2015

@author: montoya
"""

import numpy as np
import propagation
#########################################
""" PROPAGATIONS OUTPUT """
#########################################

def get_Zg (self, X):
    Z = propagation.get_Z (X, self.Wg, self.bg)
#    print Z.shape
    return Z
    
def get_G (self, Xg):
    # H = nSa x nH  Output of every Hidden neuron to every Sample
    Z = self.get_Zg(Xg)
    G = self.fg(Z)       # Apply the transfer function
#    print H
    self.G = G          # Output Hidden Matrix
    return G

def get_Htotal (self, X, Xg):
    # H = nSa x nH  Output of every Hidden neuron to every Sample
    H = self.get_H(X)
    G = self.get_G(Xg)       # Apply the transfer function
    
    Htotal = np.concatenate((H,G), axis = 1)
#    print "Htotal size: " + str(Htotal.shape)

    self.Htotal = Htotal          # Output Hidden Matrix
    return Htotal
    
#########################################
""" OUTPUT FUNCTION OBTAINING """
#########################################
def get_Zo (self, X, Xg):
    H_total = self.get_Htotal (X, Xg)
    Zo = propagation.get_Z (H_total, self.Wo, self.bo)
#    print Zo
    return Zo
    
def get_O (self, X, Xg):
    Zo = self.get_Zo(X, Xg)
    O = self.fo(Zo)
    self.O = O
#        print O
    return O

def propNoise(self,G, param):
    # Funtion that generates the noise for the propagation of the G neurons.
    # It was designed to avoid overfitting when propagating previous outputs.
    distr = param[0] 
    param = param[1:3]
    
    if (distr == "uniform"):
        noise = np.random.uniform(param[0],param[1],G.shape)

    elif (distr == "normal"):
        noise = np.random.normal(param[0],param[1],G.shape)
   
    return noise
    
    
    