# -*- coding: utf-8 -*-
"""
Created on Tue Mar 24 01:31:58 2015

@author: montoya
"""

import numpy as np

import settings as settings 
import propagation as prop
import init_centers as IC
import interface as interface

import ELM as ELM

from math_func import * # This way we import the functions directly
import paramClasses as paC

class CRBF:
    
    def __init__(self, nC = 25,
                 fb = "guassian",
                 fo = "tanh",
                 initialization = "MSE",
                 beta = 1):
                     
        self.nC = nC        # Number of centers 
        self.centers = []   # Centers of the neurons
        self.beta = beta    # Beta
        
        self.set_activation_func(fb,fo)
        
    ####################################
    """ INPUT PARAMETER FUNCTIONS """
    ###################################

    set_Train = settings.set_Train
    set_Val = settings.set_Val

    set_nC = settings.set_nC
    
    set_activation_func = settings.set_activation_func
    set_errFunc = settings.set_errFunc
    
    set_trainigAlg = settings.set_trainigAlg
    set_initCenters = settings.set_initCenters
    set_regularization = settings.set_regularization

    #########################################
    """ PROPAGATIONS OUTPUT """
    #########################################
    get_G = prop.get_G

    #########################################
    """ OUTPUT FUNCTION OBTAINING """
    #########################################
    get_Zo = prop.get_Zo
    get_O = prop.get_O

    ####################################
    """ CENTERS INITILIZATION """
    ###################################
    
    random_samples_centers = IC.random_samples_centers
    K_means = IC.K_means
    
    def init_Centers(self):
        if (self.initCenters.centersInit == "randomSamples"):
            self.random_samples_centers(self.initCenters.param)
        elif (self.initCenters.centersInit == "Kmeans"):
            self.K_means(self.initCenters.param)

###########################################################################################################################
######################################### TRAINING ALGORITHMS !!  ##########################################################
###########################################################################################################################

    #########################################
    """ ELM algorithm """
    #########################################
    ELM_train = ELM.ELM_train
    ELM_validate = ELM.ELM_validate


    def train (self, D = []):     # D is the Deemfasis fot Boosting
        
        # Adapt the labels so that they are correct (-1 or 0 and transform multivariate if needed)
        self.Ytrain = paC.adapt_labels(self.Ytrain, mode = self.outMode)
        self.Yval = paC.adapt_labels(self.Yval, mode = self.outMode )
        
        # Check the training algorithm and pass it with its parameters.
        # D is the dehenfasis vector, distribution of samples probabilities.
        if (self.trainingAlg.trAlg == "ELM"):
            self.ELM_train(self.trainingAlg.param, D)
               
###########################################################################################################################
######################################### sklearn INTEFACE !!  ##########################################################
###########################################################################################################################

    fit = interface.fit
    predict_proba = interface.predict_proba
    soft_out = interface.soft_out
    predict = interface.predict
    score = interface.score
        
###########################################################################################################################
######################################### visual SHIT !!  ##########################################################
###########################################################################################################################
 
    def set_visual (self, param = [0]):
        # Function sets the visualization parameters for obtaining the evolution
        # of intermediate results.
        self.visual = param;

