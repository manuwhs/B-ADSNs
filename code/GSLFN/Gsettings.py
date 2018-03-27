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

def set_GTrain (self, GXtrain):
    # Xtrain = M (N_samples, Ndim)
    # Ytrain = M (N_samples, Noutput)
    # Ytrain is expected to be in the form  Ytrain_i = [-1 -1 ··· 1 ··· -1 -1] for multiclass
    # This function MUST be called, ALWAYS

    self.GXtrain = GXtrain

    shape_Gtrain = GXtrain.shape 
    self.nIG = shape_Gtrain[1]      # Number of input dimensions
    
    
def set_GVal (self, GXval):
    self.GXval = GXval

def set_GTest (self, GXtest):
    self.GXtest = GXtest
    
def set_nG (self, nG):
    self.nG = nG

def set_Gactivation_func (self, fg = "tanh"):
    
    if (fg == "sigmoid"):
        self.fg = sigm
        self.dfg = dsigm
        
    if (fg == "tanh"):
        self.fg = tanh
        self.dfg = dtanh

    if (fg == "linear"):
        self.fg = linear
        self.dfg = dlinear

    if (fg == "zero"):
        self.fg = zero
        self.dfg = zero
        