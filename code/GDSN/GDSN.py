# -*- coding: utf-8 -*-
"""
Created on Tue Mar 24 01:31:58 2015

@author: montoya
"""

import numpy as np
import matplotlib.pyplot as plt

import copy as copy
import SLFN as SLFN

import settings as settings
import Dsettings as Dsettings
import interface as interface
import GDpropagation as GDprop
import Gpropagation as Gprop

import training as tr
import GDtraining as GDtr

class CGDSN:
    # Deep Stacked Network, based on SLFN base learner that is stacked.
    def __init__(self, nL = 5, nP = 1,
                visual = [0], CV = 1, Nruns = 1, InitRandomSeed = -1):
                    
        # nL: Number of layers 
        # nP: Number of previous outputs used in the nextself.nL = nL
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
    get_O = GDprop.get_O
    propNoise = Gprop.propNoise
    
    train_once = GDtr.train_once
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
    output_stuff = GDtr.output_stuff
    
    def set_visual (self, param = [0]):
        # Function sets the visualization parameters for obtaining the evolution
        # of intermediate results.
        self.visual = param;
        
