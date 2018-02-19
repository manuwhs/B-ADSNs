# -*- coding: utf-8 -*-
"""
Created on Tue Mar 24 01:31:58 2015

@author: montoya
"""

import numpy as np
import interface as interface
import settings as settings
import Dsettings as Dsettings
import Dpropagation as Dprop

import Dtraining as Dtr
import training as tr


class CDSN:
    # Deep Stacked Network, based on SLFN base learner that is stacked.
    def __init__(self, nL = 5, nP = 1,
                visual = [0], CV = 1, Nruns = 1, InitRandomSeed = -1):
                    
        # nL: Number of layers 
        # nP: Number of previous outputs used in the next
  
        self.nL = nL
        self.nP = nP
        
        self.set_visual(visual)
        self.CV = CV
        self.Nruns = Nruns
        self.D_flag = 0 # Flag that tells us if there is a distribution of samples
        
        self.InitRandomSeed = InitRandomSeed
        self.RandomSeed = np.zeros((self.Nruns,1),dtype = int) # Each run has a random seed

        self.Xtest = []  # If they stay like this we dont use them. Xtrain and Xval are mandatory
        self.Ytest = []
        self.D_flag = 0 # Flag that tells us if there is a distribution of samples
        

    set_Train = settings.set_Train
    set_Val = settings.set_Val
    set_Test = settings.set_Test
    
    set_nL = Dsettings.set_nL
    set_nP = Dsettings.set_nP
    set_Base_Layer = Dsettings.set_Base_Layer

    
    #########################################
    """ OUTPUT FUNCTION OBTAINING """
    #########################################
    get_O = Dprop.get_O
    fi = Dtr.fi                    ### FUNCION PROPAGACION WholeO ###  
    
    train_once = Dtr.train_once
    train = tr.train
    train_CV = tr.train_CV
    
###########################################################################################################################
######################################### sklearn INTEFACE !!  ##########################################################
###########################################################################################################################

    fit = interface.fit
    predict_proba = interface.predict_proba
    soft_out = interface.soft_out
    predict = interface.predict
    score = interface.score
    soft_error = interface.soft_error
    manage_results = interface.manage_results

    instant_score = interface.instant_score
    output_stuff = Dtr.output_stuff
    check_stop_L = Dtr.check_stop_L
    
    def set_visual (self, param = [0]):
        # Function sets the visualization parameters for obtaining the evolution
        # of intermediate results.
        self.visual = param;


class DSN_visual:
    
    # Class with the important parametes of the NN for results and conclussions
    def __init__(self, verbose = 0,
                 store_layers_scores = 0,
                       store_layers_soft_error = 0,
                       plot_results_layers = 0
                       ):
                           
        self.verbose = verbose
        self.plot_results_layers = plot_results_layers
        
        if (plot_results_layers == 1):
            self.store_layers_scores = 1
            self.store_layers_soft_error = 1
        
        else:
            self.store_layers_scores = store_layers_scores
            self.store_layers_soft_error = store_layers_soft_error
