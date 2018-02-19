# -*- coding: utf-8 -*-
"""
Created on Tue Mar 24 01:31:58 2015

@author: montoya
"""

import numpy as np

import Bfunc as Bf

import settings
import interface
import training as tr
import Btraining as Btr
class CBoosting:
    
    def __init__(self, T = 5, alg = "RealAdaBoost",
                 visual = [0], CV = 1, Nruns = 1, InitRandomSeed = -1):
                 
        self.set_T(T)
        self.alphas = np.ones((T,1))
        self.alg = alg
        
        # For output obtaining a viewing
        self.set_visual(visual)
        self.CV = CV
        self.Nruns = Nruns
        
        self.InitRandomSeed = InitRandomSeed
        self.RandomSeed = np.zeros((self.Nruns,1),dtype = int) # Each run has a random seed
        
        self.Xtest = []  # If they stay like this we dont use them. Xtrain and Xval are mandatory
        self.Ytest = []
        self.D = []
        
    set_Classifier = Bf.set_Classifier

    get_O = Bf.get_O
    
    set_Train = settings.set_Train
    set_Val = settings.set_Val
    set_Test = settings.set_Test
    
    set_T = Bf.set_T
    set_stopCrit = settings.set_stopCrit
    train = Btr.train
    train_CV = tr.train_CV
    train_once = Btr.train_once
    
###########################################################################################################################
#########################################  Obtain resulting network ##########################################################
###########################################################################################################################

    get_SLFN = Bf.get_SLFN
    output_stuff = Bf.output_stuff
###########################################################################################################################
######################################### sklearn INTEFACE !!  ##########################################################
###########################################################################################################################
    instant_score = interface.instant_score
    predict_proba = Bf.predict_proba
    predict = Bf.predict
    score = Bf.score
    
    def set_visual (self, param = [0]):
        # Function sets the visualization parameters for obtaining the evolution
        # of intermediate results such as:
        #   - Evolution of the error with the epochs
        self.visual = param;
