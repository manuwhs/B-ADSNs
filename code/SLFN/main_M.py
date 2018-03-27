# -*- coding: utf-8 -*-
"""
Created on Tue Mar 24 01:31:58 2015

@author: montoya
"""

import numpy as np
import matplotlib.pyplot as plt
from sklearn import preprocessing

import scipy.io
import paramClasses as paC
# Import own libraries

import SLFN 
import SLFN_M

plt.close('all')

#%% Load data
AVIRIS_data = 0;

if (AVIRIS_data == 1):
    data = np.loadtxt("../data/AVIRIS_dataset/data.txt")
    labels = np.loadtxt("../data/AVIRIS_dataset/labels.txt")
    names = np.loadtxt("../data/AVIRIS_dataset/names.txt", dtype=np.str)

    #%% Remove noisy bands
    dataR1 = data[:,:103]
    dataR2 = data[:,108:149]
    dataR3 = data[:,163:219]
    dataR = np.concatenate((dataR1,dataR2,dataR3),axis=1)
    
    #%% Exclude background class
    dataR = dataR[labels!=0,:]
    labelsR = labels[labels!=0]
    labelsR = labelsR - 1  # So that classes start at 1
    #%% Split data in training and test sets
    train_ratio = 0.2
    rang = np.arange(np.shape(dataR)[0],dtype=int) # Create array of index
    np.random.seed(0)
    rang = np.random.permutation(rang)        # Randomize the array of index
    
    Ntrain = round(train_ratio*np.shape(dataR)[0])    # Number of samples used for training
    Ntest = len(rang)-Ntrain                  # Number of samples used for testing
    
    Xtrain = dataR[rang[:Ntrain]]
    Xtest = dataR[rang[Ntrain:]]
    Ytrain = labelsR[rang[:Ntrain]]
    Ytest = labelsR[rang[Ntrain:]]
    

#################################################################
#################### DATA PREPROCESSING #########################
#################################################################
    
#%% Normalize data
scaler = preprocessing.StandardScaler().fit(Xtrain)
Xtrain = scaler.transform(Xtrain)            
Xtest = scaler.transform(Xtest)       
        
#################################################################
#################### Neural Net Using #########################
#################################################################

nH = 20;

mySLFN = SLFN.CSLFN (nH = nH, fh = "tanh", fo = "tanh")

# WEIGHT INITIALIZATION
initDistrib = paC.Init_Distrib("default", ["uniform",-1,1])# Define initialization
#initDistrib = SLFN.Init_Distrib("default", ["normal",0,1])# Define initialization
mySLFN.set_initDistrib (initDistrib)                        # Set the initialization

BP_F = 0
ELM_F = 0
BMBP_F = 0
FT_F = 1

# DEFINE TRAINING ALGORITHM
if (BP_F == 1):
    # Step and number of epochs
    trainingAlg = paC.Training_Alg("BP",[300, 0.0003 ])

if (BMBP_F == 1):
    # Step and number of epochs
    trainingAlg = paC.Training_Alg("BMBP",[1500, 0.00008])

if (ELM_F == 1):
    trainingAlg = paC.Training_Alg("ELM",["bias"])

if (FT_F == 1):
    trainingAlg = paC.Training_Alg("FT",[1000, 0.0015, "bias","normal"])
    
#    nHs = range (10, 200,5)
#    mySLFN.ELM_validate(nHs, n_iter = 10)

mySLFN.set_trainigAlg(trainingAlg)     # Set the trainig algorithm

#####################################################################
####################### Multiclass Generalization ####################
#####################################################################

SLFN_F = 1
SLFN_M_F = 0

mySLFN.set_visual([1])

if (SLFN_F  == 1):

    Ytrain_m =  paC.get_labels(Ytrain, mode = -1)
    Ytest_m  = paC.get_labels(Ytest, mode = -1)
    
    mySLFN.set_Train (Xtrain, Ytrain_m)
    mySLFN.set_Val (Xtest, Ytest_m)
    
    mySLFN.init_Weights()                    # Init weights randomly
    mySLFN.train()
    


    score = mySLFN.score(Xtrain, Ytrain_m)
    print "Training Score: " + str(score)
    
    score = mySLFN.score(Xtest, Ytest_m)
    print "Test Score: " + str(score)


if (SLFN_M_F  == 1):
    mySLFN_M = SLFN_M.CSLFN_M(mySLFN)
    
    Ytrain_m =  paC.get_labels(Ytrain, mode = -1)
    Ytest_m  = paC.get_labels(Ytest, mode = -1)
    
    mySLFN_M.set_Train (Xtrain, Ytrain_m)
    mySLFN_M.set_Val (Xtest, Ytest_m)
    mySLFN_M.train()
    
    score = mySLFN_M.score(Xtrain, Ytrain_m)
    print "Training Score: " + str(score)
    
    score = mySLFN_M.score(Xtest, Ytest_m)
    print "Test Score: " + str(score)


