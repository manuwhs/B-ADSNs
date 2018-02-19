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

import numpy as np
import GSLFN 
import paramClasses as paC

plt.close('all')

#################################################
################# LOAD DATASET ##################
#################################################

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
#################### NEURAL NETWORM PARAMETERS ##################
#################################################################
if ((Cluster_exec == 1)|(Console_exec == 1)):
    nH = int(param[0])
else:
    nH = 3

nG = 1

fh_name = "tanh"
fo_name = "linear"
fg_name = "linear"

errFunc = "MSE"

Nruns = 1
CV = 1
InitRandomSeed = -1

GSLFN_visual = [1]
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
    trainingAlg = paC.Training_Alg("BP",[500,0.0003, 0.2])
    
if (BMBP_F == 1):
    # Number of epochs, step and number of partitions
    trainingAlg = paC.Training_Alg("BMBP",[200, 0.001, 10, 3])

if (ELM_F == 1):
    trainingAlg = paC.Training_Alg("ELM",["bias"])
    
if (ELMT_F == 1):
    trainingAlg = paC.Training_Alg("ELMT",[10, 0.0005, 10, "bias"])

if (LDA_F == 1):
    trainingAlg = paC.Training_Alg("LDA", [])

if (LDAT_F == 1):
    trainingAlg = paC.Training_Alg("LDAT",[200, 0.0005, 10])


#################################################################
#################### Neural Net Using #########################
#################################################################

# Create cuadratic expansion
GXtrain = np.power(Xtrain,2)
GXtest = np.power(Xtest,2)

myGSLFN = GSLFN.CGSLFN (nH = nH,
                         fh = fh_name,
                         fo = fo_name,
                         fg = fg_name,
                         errFunc = errFunc,
                         CV = CV,
                         Nruns = Nruns,
                         InitRandomSeed = InitRandomSeed)
                         
myGSLFN.set_Train (Xtrain, Ytrain)
myGSLFN.set_GTrain (GXtrain)

myGSLFN.set_Val (Xtest, Ytest)  # If CV > 1, it is not used.
myGSLFN.set_GVal (GXtest)

myGSLFN.set_Test (Xtest, Ytest)
myGSLFN.set_GTest (GXtest)
    
myGSLFN.set_initDistrib (initDistrib)        # Set the initialization
myGSLFN.set_trainigAlg(trainingAlg)          # Set the trainig algorithm

myGSLFN.set_stopCrit(stopCriterion)          # Set the Stop Criterion
myGSLFN.set_regularization(regularization)   # Set regularization

myGSLFN.set_visual(GSLFN_visual)                      # Visualization of output

myGSLFN.train()                              # Train the algorithm
myGSLFN.output_stuff(output_folder,[myGSLFN.nH])            # The name os the file will be the parameters to validate


print "Training Score: " + str(myGSLFN.TrError)

print "Validation Score: " + str(myGSLFN.ValError)

print "Test Score: " + str(myGSLFN.TstError)

#print "Test Score: " + str(mySLFN.score(Xtest, Ytest))
