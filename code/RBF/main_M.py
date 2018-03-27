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

import RBF 

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

#Xtrain = Xtrain[:,30:80]
#Xtest = Xtest[:,30:80]

#################################################################
#################### Neural Net Using #########################
#################################################################

# Number of centers
nC = 250;

myRBF = RBF.CRBF (nC = nC,
                 fb = "guassian",
                 fo = "tanh",
                 beta = 0.1);
                 
myRBF.set_Train (Xtrain, Ytrain)
myRBF.set_Val (Xtest, Ytest)


# CENTRE INITIALIZATION
initCenters = paC.Init_Centers("randomSamples") # Define initialization of centres
#initCenters = paC.Init_Centers("Kmeans", [300,10,"nosplit"]) # Define initialization of centres

myRBF.set_initCenters (initCenters)                        # Set the initialization
                  # Set the initialization

BP_F = 0
ELM_F = 1
BMBP_F = 0
FT_F = 0

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
    trainingAlg = paC.Training_Alg("FT",[7000, 0.00015, "bias","normal"])
    
#    nHs = range (10, 200,5)
#    mySLFN.ELM_validate(nHs, n_iter = 10)

myRBF.set_trainigAlg(trainingAlg)     # Set the trainig algorithm

#####################################################################
####################### Multiclass Generalization ####################
#####################################################################


myRBF.set_visual([1])


Ytrain_m =  paC.get_labels(Ytrain, mode = -1)
Ytest_m  = paC.get_labels(Ytest, mode = -1)

myRBF.set_Train (Xtrain, Ytrain_m)
myRBF.set_Val (Xtest, Ytest_m)

myRBF.set_trainigAlg(trainingAlg)     # Set the trainig algorithm
myRBF.set_visual([1])                 # Set verbose options

myRBF.init_Centers()                  # Initialize
myRBF.train()                         # Train the algorithm

#nCs = range (10, 100,5)
#myRBF.ELM_validate(nCs,param = ["bias"], n_iter = 10)

score = myRBF.score(Xtrain, Ytrain_m)
print "Training Score: " + str(score)

score = myRBF.score(Xtest, Ytest_m)
print "Test Score: " + str(score)

