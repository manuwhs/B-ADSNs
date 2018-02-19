# -*- coding: utf-8 -*-
"""
Created on Tue Mar 24 01:31:58 2015

@author: montoya
"""

import numpy as np

import init_weights as IW
import propagation as prop
import settings as settings 
import interface as interface


import BP as BP
import BMBP as BMBP
import ELMTuning as ELMT
import LDATuning as LDAT
import training as tr



class CSLFN:
    
    def __init__(self, nH = 25, fh = "tanh", fo = "tanh", errFunc = "MSE", 
                 trainingAlg = [], initDistrib = [], regularization = [],
                 visual = [0], CV = 1, Nruns = 1, InitRandomSeed = -1):
        #  MSE: Mean Square Error
        #  Cross-Entropy: NLL of probability (output)
        self.nH = nH                        # Number of hidden neurons 
        self.set_activation_func(fh,fo);    # Activation functions of hidden and output neurons
        self.set_errFunc(errFunc)           # Error Function (some algorihms impose this function)
  
        self.set_trainigAlg(trainingAlg)               # Training Algorithm
        self.set_initDistrib (initDistrib)  # Distribution for the initialization
        self.set_regularization (regularization)
        
        # For output obtaining a viewing
        self.set_visual(visual)
        self.CV = CV
        self.Nruns = Nruns
        
        self.InitRandomSeed = InitRandomSeed
        self.RandomSeed = np.zeros((self.Nruns,1),dtype = int) # Each run has a random seed
        
        self.Xtest = []  # If they stay like this we dont use them. Xtrain and Xval are mandatory
        self.Ytest = []
        self.D_flag = 0 # Flag that tells us if there is a distribution of samples
        
    ####################################
    """ INPUT PARAMETER FUNCTIONS """
    ###################################

    set_Train = settings.set_Train
    set_Val = settings.set_Val
    set_Test = settings.set_Test
    
    set_nH = settings.set_nH
    set_D = settings.set_D
    
    set_activation_func = settings.set_activation_func
    set_errFunc = settings.set_errFunc
    
    set_trainigAlg = settings.set_trainigAlg
    set_initDistrib = settings.set_initDistrib
    set_regularization = settings.set_regularization
    set_stopCrit = settings.set_stopCrit
    
    #########################################
    """ PROPAGATIONS OUTPUT """
    #########################################
    get_Zh = prop.get_Zh  
    get_H = prop.get_H

    #########################################
    """ OUTPUT FUNCTION OBTAINING """
    #########################################
    get_Zo = prop.get_Zo
    get_O = prop.get_O
    
    evaluate_stop = prop.evaluate_stop
    ####################################
    """ WEIGHTS INITILIZATION """
    ###################################
    
    init_Weights = IW.init_Weights
    init_default = IW.init_default   # Default initialization
    init_deep_learning = IW.init_deep_learning   # DeepLearning initialization

###########################################################################################################################
######################################### TRAINING ALGORITHMS !!  ##########################################################
###########################################################################################################################
    #########################################
    """ BP algorithm """
    #########################################
    BP_train = BP.BP_train
    BP_validate = BP.BP_validate

    #########################################
    """ BMBP algorithm """
    #########################################
    BMBP_train = BMBP.BMBP_train
    
    #########################################
    """ ELM algorithm """
    #########################################
    ELM_train = ELMT.ELM_train
    ELMT_train = ELMT.ELMT_train
    
    #########################################
    """ LDAT algorithm """
    #########################################
    LDA_train = LDAT.LDA_train
    LDAT_train = LDAT.LDAT_train
    
    #########################################
    """ TRAINING """
    #########################################
    train = tr.train
    train_CV = tr.train_CV
    train_once = tr.train_once
    
###########################################################################################################################
######################################### sklearn INTEFACE !!  ##########################################################
###########################################################################################################################

    fit = interface.fit
    predict_proba = interface.predict_proba
    soft_out = interface.soft_out
    predict = interface.predict
    score = interface.score
    instant_score = interface.instant_score
    soft_error = interface.soft_error
    get_MSE = interface.get_MSE
    manage_results = interface.manage_results
    output_stuff = interface.output_stuff
    
###########################################################################################################################
######################################### visual SHIT !!  ##########################################################
###########################################################################################################################
 
    def set_visual (self, param = [0]):
        # Function sets the visualization parameters for obtaining the evolution
        # of intermediate results such as:
        #   - Evolution of the error with the epochs
        self.visual = param;
    

