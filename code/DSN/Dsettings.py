# -*- coding: utf-8 -*-
"""
Created on Tue Mar 24 01:31:58 2015

@author: montoya
"""

import numpy as np

from math_func import * # This way we import the functions directly

####################################
""" INPUT PARAMETER FUNCTIONS """
###################################

def set_Base_Layer (self, layer):
    # This functions sets the Base Layer, a SLFN network
    # This classifier must have all its personal parameters set and the methods:
    # classifier.fit(Xtrain, Ytrain, W)
    # classifier.soft_out(X)
    self.base_layer = layer
    self.layers = [];             # List of the layers

    self.outMode = layer.outMode
def set_nL (self, nL):
    # Sets the number of layers of the network.

    self.nL = nL

def set_nP (self, nP):
    # Sets the number of previous outputs propagated.

    self.nP = nP
    