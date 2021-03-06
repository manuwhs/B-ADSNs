# -*- coding: utf-8 -*-
"""
Created on Tue Mar 24 01:31:58 2015

@author: montoya
"""

#################################################
############ HEADING FOR WORKING ################
#################################################

Cluster_exec = 0
Spyder_exec = 1
Console_exec = 0

if (Cluster_exec == 1): # If executed using command line from root folder.
    import os
    import sys
    ##################
    """ Aparently os and sys references are from the dir where you execute
    and "import" acts from the directory the file executes ! Hijo de puta """
    ##################
    # That is why the first thing we do is importing the main folder so
    # that we can import "clusterizer" 
    
    sys.path.append(os.path.abspath(''))  
    import clusterizing as clus
    
    ## Param contains the str parameters
    param = clus.clusterize()
    print param

    # Configure cluster folders for data input and output
    output_folder = "/export/clusterdata/mmontoya/Results/"
    dataset_folder = "/export/clusterdata/mmontoya/data/"
    
if (Spyder_exec == 1): # If it is executed from spyder
    import os
    import sys
    whole_path = os.path.abspath('')
    folder_name = os.path.basename(whole_path)
    root_path = whole_path.split(folder_name)[0]
    sys.path.append(root_path)   # Include the root path into the "import" paths

    import import_folders
    import_folders.imp_folders(root_path)

    output_folder = "../Results/"
    dataset_folder = "../data/"

if (Console_exec == 1): # If executed using command line from root folder.
    import os
    import sys
    ##################
    """ Aparently os and sys references are from the dir where you execute
    and "import" acts from the directory the file executes ! Hijo de puta """
    ##################
    # That is why the first thing we do is importing the main folder so
    # that we can import "clusterizer" 
    
    sys.path.append(os.path.abspath(''))  
    import clusterizing as clus
    
    ## Param contains the str parameters
    param = clus.clusterize()
    print param

    # Configure cluster folders for data input and output
    output_folder = "../Results/"
    dataset_folder = "../data/"
    

#################################################
############ HEADING FOR WORKING ################
#################################################

import matplotlib.pyplot as plt
from sklearn import preprocessing
import scipy.io
import Boosting as  Boosting
import SLFN 
import paramClasses as paC

import numpy as np
plt.close('all')


#################################################
################# LOAD DATASET ##################
#################################################
import dataset_preprocessing as dp
print dataset_folder
X,Y = dp.abalone_dataset(dataset_folder + "/abalone/abalone.data")
#Xtrain, Xtest, Ytrain, Ytest = dp.obtain_train_test (X, Y, train_ratio = 0.6)


mat = scipy.io.loadmat(dataset_folder +'abalone.mat')
#mat = scipy.io.loadmat(dataset_folder +'ionosfera.mat')
#mat = scipy.io.loadmat(dataset_folder +'kwok.mat')
#mat = scipy.io.loadmat(dataset_folder +'ripley.mat')

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
#################### BOOSTING PARAMETERS ##################
#################################################################
        
T = 1000
    
Nruns = 1  # These are 1 for Boosting !!
CV = 1
InitRandomSeed = -1

######## BOOSTING STOP CRITERION  ########
sCBust = paC.Stop_Criterion("Nmax",[])

#################################################################
#################### NEURAL NETWORM PARAMETERS ##################
#################################################################
if ((Cluster_exec == 1)|(Console_exec == 1)):
    nH = int(param[0])
else:
    nH = 6

fh_name = "tanh"
fo_name = "tanh"
errFunc = "MSE"
visual = 0

print "Number of neurons: " + str(nH)

######## WEIGHT INITIALIZATION  ########
initDistrib = paC.Init_Distrib("default", ["uniform",-1,1])
#initDistrib = paC.Init_Distrib("default", ["normal",0,1])
#initDistrib = paC.Init_Distrib("deepLearning", ["DL1"])

######## STOP CRITERION  ########
stopCriterion = paC.Stop_Criterion("Nmax",[])

########  REGULARIZATION   ########
regularization = paC.Regularization("L2",[0.000005])

######## TRAINING ALGORITHM  ########

BP_F = 0
BMBP_F = 1

ELM_F = 0
ELMT_F = 0

LDA_F = 0
LDAT_F = 0

if (BP_F == 1):
    # Number of epochs, step and mommentum
    trainingAlg = paC.Training_Alg("BP",[100,0.001, 0.0])
    
if (BMBP_F == 1):
    # Number of epochs, step and number of partitions
    trainingAlg = paC.Training_Alg("BMBP",[100, 0.001, 5])

if (ELM_F == 1):
    trainingAlg = paC.Training_Alg("ELM",["bias"])
    
if (ELMT_F == 1):
    trainingAlg = paC.Training_Alg("ELMT",[20, 0.0005, 10, "bias"])

if (LDA_F == 1):
    trainingAlg = paC.Training_Alg("LDA", [])

if (LDAT_F == 1):
    trainingAlg = paC.Training_Alg("LDAT",[200, 0.0005, 10])

######## CREATE NETWORK AND SET IT UP  ########
#mySLFN = SLFN.CSLFN (nH = nH, fh = "sigmoid", fo = "sigmoid", errFunc = "CE")

mySLFN = SLFN.CSLFN (nH = nH,
                     fh = fh_name,
                     fo = fo_name,
                     errFunc = errFunc,
                     CV = 1,
                     Nruns = 1,
                     InitRandomSeed = -1)

mySLFN.set_Train (Xtrain, Ytrain)
mySLFN.set_Val (Xtest, Ytest)    # If CV > 1, it is not used.
mySLFN.set_Test (Xtest, Ytest)   # For having a same testing subset always
                                 # Not used during training 

mySLFN.set_initDistrib (initDistrib)        # Set the initialization
mySLFN.set_trainigAlg(trainingAlg)          # Set the trainig algorithm

mySLFN.set_stopCrit(stopCriterion)          # Set the Stop Criterion
mySLFN.set_regularization(regularization)   # Set regularization

mySLFN.set_visual([visual])                      # Visualization of output

#################################################################
########################## Boosting #########################
#################################################################


myBust = Boosting.CBoosting(T = T,
                            alg = "RealAdaBoost",
                            CV = CV,
                            Nruns = Nruns,
                            InitRandomSeed = InitRandomSeed); # GentleBoost  RealAdaBoost

myBust.set_Train (Xtrain, Ytrain)
myBust.set_Val (Xtest, Ytest)
myBust.set_Test (Xtest, Ytest)

myBust.set_Classifier(mySLFN);

myBust.set_stopCrit(sCBust)

myBust.train();


myBust.output_stuff(output_folder,[myBust.base_learner.nH])            # The name os the file will be the parameters to validate


print "Training Score: " + str(myBust.TrError)

print "Validation Score: " + str(myBust.ValError)

print "Test Score: " + str(myBust.TstError)

