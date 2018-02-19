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
    
    self.Wh = np.zeros((self.nI,self.nH))
    self.Wo = np.zeros((self.nH,self.nO))
    
    self.bh = np.zeros((1,self.nH))
    self.bo = np.zeros((1,self.nO))   
    
#    print self
#    print distr
#    print self.nH
    
    if (distr == "uniform"):
        self.Wh = np.random.uniform(param[0],param[1],(self.nI,self.nH))
        self.Wo = np.random.uniform(param[0],param[1],(self.nH,self.nO))
        
        self.bh = np.random.uniform(param[0],param[1],(1,self.nH))
        self.bo = np.random.uniform(param[0],param[1],(1,self.nO))       

    elif (distr == "normal"):
        self.Wh = np.random.normal(param[0],param[1],(self.nI,self.nH))
        self.Wo = np.random.normal(param[0],param[1],(self.nH,self.nO))
        
        self.bh = np.random.normal(param[0],param[1],(1,self.nH))
        self.bo = np.random.normal(param[0],param[1],(1,self.nO))       
    else:
        print "WERTHERTHEDSRTGNBSRTHBSRT"
    
#    print self.Wh
    
def init_deep_learning (self, param ):
    
    distr = param[0] 
    Ninit = param[1]

    
    if (distr == "DL1"):  # DeepLearning 1
        
        k = float(Ninit)  # Usually 6.0
        rh = np.sqrt(k/(self.nI + self.nH))
        ro = np.sqrt(k/(self.nH + self.nO))
#        print "Rh, ro " + str(rh) + "  "+ str(ro)
        
#        print str(self.nI) + " " + str(self.nH) + " " + str(self.nO)
#        print rh
#        print ro
        
        if (self.outMode == 0):  # If the functions are sigmoidal
            rh = rh * 4
            ro = ro * 4
            print "frververbe"
            
        self.Wh = np.random.uniform(-rh ,rh,(self.nI,self.nH))
        self.Wo = np.random.uniform(-ro,ro,(self.nH,self.nO))
        
        self.bh = np.random.uniform(-rh ,rh,(1,self.nH))
        self.bo = np.random.uniform(-ro,ro,(1,self.nO))         

#        print self.Wo
        
    elif (distr == "DL2"): # DeepLearning 2
        
        rh = 1 / np.sqrt((self.nI))
        ro = 1 / np.sqrt((self.nH))
        
            
        self.Wh = np.random.uniform(-rh ,rh, (self.nI,self.nH))
        self.Wo = np.random.uniform(-ro,ro, (self.nH,self.nO))
        
        self.bh = np.random.uniform(-rh ,rh, (1,self.nH))
        self.bo = np.random.uniform(-ro,ro, (1,self.nO))  

    elif (distr == "Anibal"):  # DeepLearning 1

        self.Wh = np.random.uniform(-1 ,1,(self.nI,self.nH))
        self.bh = np.random.uniform(0 ,0,(1,self.nH))
        
        self.Wo = np.random.uniform(-1,1,(self.nH,self.nO))
        self.bo = np.random.uniform(0,0,(1,self.nO))         

        # Now we normalize Zo with (Wo,bo), so that O is [-0.8,0.8]
        
        # For every hidden neuron:
        
        Ni, Nh = self.Wh.shape
        k = 4
#        print Ni,Nh
        for i in range(Nh):
            Zh = self.Xtrain.dot(self.Wh[:,i])
            std = np.std(Zh)
            mean = np.mean(Zh)
#            print std,mean
            self.Wh[:,i] = self.Wh[:,i] / (k*std)
            self.bh[0,i] = - mean / (k*std)       
            
        Zo = self.get_H(self.Xtrain).dot(self.Wo)
        
#        print Zo
        
        std = np.std(Zo)
        mean = np.mean(Zo)
#        print std,mean
        

        min_Zo = np.min(Zo)
        max_Zo = np.max(Zo)
        diff_Zo = max_Zo - min_Zo
#        print diff_Zo, std, mean

        self.Wo = self.Wo / (k*std)
        self.bo = 0 # - mean / (k*std)

#        Zo = self.get_Zo(self.Xtrain)
#        min_Zo = np.min(Zo)
#        max_Zo = np.max(Zo)
#        diff_Zo = max_Zo - min_Zo
#        std = np.std(Zo)
#        mean = np.mean(Zo)
#        
#        print diff_Zo, std, mean
#        print "-------------------------_"
        
#        self.Wo = self.Wo / (2*diff_Zo)
#        self.bo = - mean_Zo/ (2*diff_Zo)

    elif (distr == "Anibal2"):  # DeepLearning 1
    
        k = 6.0  # Usually 6.0
        rh = np.sqrt(k/(self.nI + self.nH))

#        print str(self.nI) + " " + str(self.nH) + " " + str(self.nO)
#        print rh
#        print ro
        
        if (self.outMode == 0):  # If the functions are sigmoidal
            rh = rh * 4
            ro = ro * 4
            print "frververbe"
            
        self.Wh = np.random.uniform(-rh ,rh,(self.nI,self.nH))
        self.bh = np.random.uniform(-rh ,rh,(1,self.nH))


        self.Wo = np.random.uniform(-1,1,(self.nH,self.nO))
        self.bo = np.random.uniform(0,0,(1,self.nO))         

        # Now we normalize Zo with (Wo,bo), so that O is [-0.8,0.8]
        
        # For every hidden neuron:
        
        Ni, Nh = self.Wh.shape
        k = 4
        Zo = self.get_H(self.Xtrain).dot(self.Wo)
        
        std = np.std(Zo)
        mean = np.mean(Zo)
#        print std,mean
        
        min_Zo = np.min(Zo)
        max_Zo = np.max(Zo)


        self.Wo = self.Wo / (k*std)
        self.bo = 0 # - mean / (k*std)
    else:
        print "WRONG INITIALIZATION DISTRIBUTION"




            