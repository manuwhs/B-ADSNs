# -*- coding: utf-8 -*-
"""
Created on Tue Mar 24 01:31:58 2015

@author: montoya
"""

import numpy as np

import propagation as prop
import settings as settings 

import Ginit_weights as GIW
import Gpropagation as Gprop
import Gsettings as Gsettings 
import Ginterface as Ginterface
import interface as interface
import training as tr

import GELMTuning as GELMT
import GLDATuning as GLDAT
import GBP as GBP
import GBMBP as GBMBP
import Gtraining as Gtr


class CGSLFN:
    
    # SLFN that has two different sets of hidden neurons:
    #  1- Normal one for the input
    #  2- The one for the output of the previous network (God hidden)
    # Both layers then connect to the outer layer. The two sets of hidden neurons
    # do not share any commun input.
    # Now we have the parameters:

    # nH = Number of normal hidden neurons
    # nG = Number of Gods neurons
    # nI = Number of inputs of the normal hidden unit
    # nIG = Number of inputs of the hiddel Gods layer

    # We can make it generating 2 SLFNS but of course we have to modify all
    # the training and initializaiton algorithms.

    def __init__(self, nH = 25, nG = 4,errFunc = "MSE", 
                 fh = "tanh", fo = "tanh", fg = "linear", 
                 trainingAlg = [], initDistrib = [], regularization = [],
                 visual = [0], CV = 1, Nruns = 1, InitRandomSeed = -1):
                     
                     
        self.nG = nG                  
        self.nH = nH                        # Number of hidden neurons 
        self.set_activation_func(fh,fo);    # Activation functions of hidden and output neurons
        self.set_Gactivation_func(fg);      # Activation functions of hidden and output neurons
        self.set_errFunc(errFunc)           # Error Function (some algorihms impose this function)
  
        self.set_trainigAlg(trainingAlg)    # Training Algorithm
        self.set_initDistrib (initDistrib)  # Distribution for the initialization

        self.set_regularization (regularization)
        
        self.set_visual(visual)
        self.CV = CV
        self.Nruns = Nruns
        self.D_flag = 0 # Flag that tells us if there is a distribution of samples
        
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
    
    # Special part of the G structure
    set_GTrain = Gsettings.set_GTrain
    set_GVal = Gsettings.set_GVal
    set_GTest = Gsettings.set_GTest
    set_Gactivation_func = Gsettings.set_Gactivation_func
    set_nG = Gsettings.set_nG
    
    #########################################
    """ PROPAGATIONS OUTPUT """
    #########################################
    get_Zh = prop.get_Zh
    get_H = prop.get_H

    get_Zg = Gprop.get_Zg
    get_G = Gprop.get_G
    get_Htotal = Gprop.get_Htotal
    #########################################
    """ OUTPUT FUNCTION OBTAINING """
    #########################################
    get_Zo = Gprop.get_Zo   # Gprop !!!!!!!
    get_O = Gprop.get_O
    propNoise = Gprop.propNoise

    evaluate_stop = prop.evaluate_stop
    ####################################
    """ WEIGHTS INITILIZATION """
    ###################################
    
    init_Weights = GIW.init_Weights
    init_default = GIW.init_default   # Default initialization
    init_deep_learning = GIW.init_deep_learning   # DeepLearning initialization
        

###########################################################################################################################
######################################### TRAINING ALGORITHMS !!  ##########################################################
###########################################################################################################################
    #########################################
    """ BP algorithm """
    #########################################
    BP_train = GBP.BP_train
    BP_validate = GBP.BP_validate

    #########################################
    """ BMBP algorithm """
    #########################################
    BMBP_train = GBMBP.BMBP_train
    
    #########################################
    """ ELM algorithm """
    #########################################
    ELM_train = GELMT.ELM_train
    ELMT_train = GELMT.ELMT_train
    
    #########################################
    """ LDAT algorithm """
    #########################################
    LDA_train = GLDAT.LDA_train
    LDAT_train = GLDAT.LDAT_train
    
    #########################################
    """ TRAINING """
    #########################################
    train = tr.train
    train_CV = Gtr.train_CV
    train_once = tr.train_once
    
    
###########################################################################################################################
######################################### sklearn INTEFACE !!  ##########################################################
###########################################################################################################################

    fit = Ginterface.fit
    predict_proba = Ginterface.predict_proba
    soft_out = Ginterface.soft_out
    predict = Ginterface.predict
    score = Ginterface.score
    
    soft_error = interface.soft_error
    get_MSE = interface.get_MSE
    instant_score = interface.instant_score

    manage_results = interface.manage_results
  
    output_stuff = Ginterface.output_stuff
###########################################################################################################################
######################################### visual SHIT !!  ##########################################################
###########################################################################################################################
 
    def set_visual (self, param = [0]):
        # Function sets the visualization parameters for obtaining the evolution
        # of intermediate results.
        self.visual = param;
    
    

        