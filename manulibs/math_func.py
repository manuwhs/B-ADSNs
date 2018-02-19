# -*- coding: utf-8 -*-
"""
Created on Tue Mar 24 01:31:58 2015

@author: montoya
"""

import numpy as np

def tanh(x):
    return np.tanh(x)

def dtanh(x):
    aux = tanh(x)
    return 1.0 - aux**2

def sigm(x):
    return 1/(1 + np.exp(-x))

def dsigm(x):
    aux = sigm(x)
    return aux * (1 - aux)

def linear(x):
    return x

def dlinear(x):
    return np.ones(x.shape)
    
def zero(x):
    return np.zeros(x.shape)