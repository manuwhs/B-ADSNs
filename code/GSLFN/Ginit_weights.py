# -*- coding: utf-8 -*-
"""
Created on Tue Mar 24 01:31:58 2015

@author: montoya
"""

import numpy as np
from math_func import * # This way we import the functions directly

def init_Weights (self):
    if (self.initDistrib.weightInit == "default"):
        self.init_default(self.initDistrib.param)
        
    if (self.initDistrib.weightInit == "deepLearning"):
        self.init_deep_learning(self.initDistrib.param)
        
def init_default (self, param ):
    # Initializes weights and bias according to the parameters nI, nH, nO
    # dist: Distribution for the initializaiton
    # param: Parameters of the distribution.
    #        Uniform ->  [ini, end]
    #        Normal   -> [Mean, Variance]
    
    distr = param[0] 
    param = param[1:3]
    
    # Normal hidden layer parameters
    self.Wh = np.zeros((self.nI,self.nH))
    self.bh = np.zeros((1,self.nH))
    
    # God's hidden layer parameters
    self.Wg = np.zeros((self.nIG,self.nG))
    self.bg = np.zeros((1,self.nG))
    
    # Output layer parameters.
    self.Wo = np.zeros((self.nH + self.nG,self.nO))
    self.bo = np.zeros((1,self.nO))   
    
    if (distr == "uniform"):
        self.Wh = np.random.uniform(param[0],param[1],(self.nI,self.nH))
        self.bh = np.random.uniform(param[0],param[1],(1,self.nH))
        
        self.Wg = np.random.uniform(param[0],param[1],(self.nIG,self.nG))
        self.bg = np.random.uniform(param[0],param[1],(1,self.nG))
        
        self.Wo = np.random.uniform(param[0],param[1],(self.nH + self.nG,self.nO))
        self.bo = np.random.uniform(param[0],param[1],(1,self.nO))       



    elif (distr == "normal"):
        self.Wh = np.random.normal(param[0],param[1],(self.nI,self.nH))
        self.bh = np.random.normal(param[0],param[1],(1,self.nH))
        
        self.Wg = np.random.normal(param[0],param[1],(self.nIG,self.nG))
        self.bg = np.random.normal(param[0],param[1],(1,self.nG))
        
        self.Wo = np.random.normal(param[0],param[1],(self.nH + self.nG,self.nO))
        self.bo = np.random.normal(param[0],param[1],(1,self.nO))       

    else:
        print "WRONG INITIALIZATION DISTRIBUTION"
        
def init_deep_learning (self, param ):
    
    distr = param[0] 
    Ninit = param[1]
#    print "FSAFDGSDFHGSDB"
#    print "INIT $$$$$$$$$$$$$$$$$$$$$$$$"
    if (distr == "DL1"):  # DeepLearning 1
        
        k = float(Ninit)  # Usually 6.0
        rh = np.sqrt(k/(self.nI + self.nH))
        ro = np.sqrt(k/(self.nH + self.nG + self.nO ))
#        print "Rh, ro " + str(rh) + "  "+ str(ro)
        
#        print str(self.nI) + " " + str(self.nH) + " " + str(self.nO)
#        print rh
#        print ro
        
        self.ro_G = ro  # For the last neuron
        # ro / (self.nH + self.nG)
        
        if (self.outMode == 0):  # If the functions are sigmoidal
            rh = rh * 4
            ro = ro * 4
            print "frververbe"
            
        self.Wh = np.random.uniform(-rh ,rh,(self.nI,self.nH ))
        self.Wo = np.random.uniform(-ro,ro,(self.nH + self.nG,self.nO))
        
        """ INIT OF THE Wo of the G euron"""
        self.Wo[self.nH:,:] = np.ones((1, self.nO))*self.ro_G  # Set it initially to 1* max(ro)
        
#        print self.Wo[self.nH:,:]
        
        self.bh = np.random.uniform(-rh ,rh,(1,self.nH))
        self.bo = np.random.uniform(-ro,ro,(1,self.nO))       
        
    self.Wg = np.ones((self.nIG,self.nG))  # No prenothing
    self.bg = np.zeros((1,self.nG))        