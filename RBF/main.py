# -*- coding: utf-8 -*-
"""
Created on Tue Mar 24 01:31:58 2015

@author: montoya
"""

import numpy as np
import matplotlib.pyplot as plt
from sklearn import preprocessing

import scipy.io

# Import own libraries

import RBF 
import paramClasses as paC

plt.close('all')


mat = scipy.io.loadmat('../data/abalone.mat')
#mat = scipy.io.loadmat('../data/ionosfera.mat')
#mat = scipy.io.loadmat('../data/kwok.mat')
#mat = scipy.io.loadmat('../data/ripley.mat')

Xtrain = mat["X_tr"]
Ytrain = mat["T_tr"]

Xtest = mat["X_tst"]
Ytest = mat["T_tst"]

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

# Number of centers
nC = 100;

myRBF = RBF.CRBF (nC = nC,
                 fb = "guassian",
                 fo = "tanh",
                 beta = 0.1);
                 
myRBF.set_Train (Xtrain, Ytrain)
myRBF.set_Val (Xtest, Ytest)


# CENTRE INITIALIZATION
#initCenters = paC.Init_Centers("randomSamples") # Define initialization of centres
initCenters = paC.Init_Centers("Kmeans", [300,10,"nosplit"]) # Define initialization of centres

myRBF.set_initCenters (initCenters)                        # Set the initialization

BP_F = 0
ELM_F = 1
BMBP_F = 0
FT_F = 0

# DEFINE TRAINING ALGORITHM
if (BP_F == 1):
    # Step and number of epochs
    trainingAlg = paC.Training_Alg("BP",[200,0.0005, 0.7])

if (BMBP_F == 1):
    # Step and number of epochs
    trainingAlg = paC.Training_Alg("BMBP",[100, 0.0005])

if (ELM_F == 1):
    trainingAlg = paC.Training_Alg("ELM",["bias"])
    
if (FT_F == 1):
    trainingAlg = paC.Training_Alg("FT",[20, 0.0005, "bias","normal"])
    

myRBF.set_trainigAlg(trainingAlg)     # Set the trainig algorithm
myRBF.set_visual([1])                 # Set verbose options

myRBF.init_Centers()                  # Initialize
myRBF.train()                         # Train the algorithm

#nCs = range (10, 100,5)
#myRBF.ELM_validate(nCs,param = ["bias"], n_iter = 10)

score = myRBF.score(Xtrain, Ytrain)
print "Training Score: " + str(score)

score = myRBF.score(Xtest, Ytest)
print "Test Score: " + str(score)

