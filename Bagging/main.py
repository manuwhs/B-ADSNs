# -*- coding: utf-8 -*-
"""
Created on Tue Mar 24 01:31:58 2015

@author: montoya
"""

import numpy as np
import matplotlib.pyplot as plt
from sklearn import preprocessing
import Bagging as  Bagging
import scipy.io

# Import own libraries

import SLFN 
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

nH = 80

mySLFN = SLFN.CSLFN (nH = nH, fh = "tanh", fo = "tanh", errFunc = "EXP")
mySLFN.set_Train (Xtrain, Ytrain)
mySLFN.set_Val (Xtest, Ytest)


# WEIGHT INITIALIZATION
initDistrib = paC.Init_Distrib("default", ["uniform",-1,1])# Define initialization
#initDistrib = paC.Init_Distrib("default", ["normal",0,1])# Define initialization

mySLFN.set_initDistrib (initDistrib)                        # Set the initialization
mySLFN.init_Weights()                                       # Initialize

BP_F = 0
ELM_F = 1
BMBP_F = 0
FT_F = 0

# DEFINE TRAINING ALGORITHM
if (BP_F == 1):
    # Step and number of epochs
    trainingAlg = paC.Training_Alg("BP",[1000,0.0003])

if (BMBP_F == 1):
    # Step and number of epochs
    trainingAlg = paC.Training_Alg("BMBP",[100, 0.0005])

if (ELM_F == 1):
    trainingAlg = paC.Training_Alg("ELM",["bias"])
    
if (FT_F == 1):
    trainingAlg = paC.Training_Alg("FT",[100, 0.0005, "bias","normal"])
    
mySLFN.set_trainigAlg(trainingAlg)     # Set the trainig algorithm

#################################################################
########################## Boosting #########################
#################################################################

myBag = Bagging.CBagging(nB = 100, classifier = mySLFN)
myBag.set_Train (Xtrain, Ytrain)
myBag.set_Val (Xtest, Ytest)

myBag.set_Classifier(mySLFN)
myBag.train();


score = myBag.score(Xtrain, Ytrain)
print "Training Score: " + str(score)

score = myBag.score(Xtest, Ytest)
print "Test Score: " + str(score)