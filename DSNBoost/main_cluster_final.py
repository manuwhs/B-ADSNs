# -*- coding: utf-8 -*-
"""
Created on Tue Mar 24 01:31:58 2015

@author: montoya
"""

#################################################
############ HEADING FOR WORKING ################
#################################################
""" SAME AS MAIN CLUSTER BUT FOR LAUNCHING THE LAST EXPERIMENTS USING THE BEST
CROSSVALIDATED PARAMETERS """

Cluster_exec = 0
Spyder_exec = 0
Console_exec = 1

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
    output_folder = "/export/clusterdata/mmontoya/ResultsNeoDSN/"
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
import DSNBoost
import SLFN 
import paramClasses as paC

plt.close('all')

#################################################
################# LOAD DATASET ##################
#################################################

mat = scipy.io.loadmat(dataset_folder +'abalone.mat')
#mat = scipy.io.loadmat(dataset_folder +'image.mat')
#mat = scipy.io.loadmat(dataset_folder +'kwok.mat')
#mat = scipy.io.loadmat(dataset_folder +'ripley.mat')

Xtrain = mat["X_tr"]
Ytrain = mat["T_tr"]

Xtest = mat["X_tst"]
Ytest = mat["T_tst"]


#X,Y = dp.abalone_dataset(dataset_folder + "/abalone/abalone.data")
#Xtrain, Xtest, Ytrain, Ytest = dp.obtain_train_test (X, Y, train_ratio = 0.6)
#################################################################
#################### DATA PREPROCESSING #########################
#################################################################
    
#%% Normalize data
scaler = preprocessing.StandardScaler().fit(Xtrain)
Xtrain = scaler.transform(Xtrain)            
Xtest = scaler.transform(Xtest)       
        
#################################################################
########################## DSNBoost PARAMETERS ########################
#################################################################
        
L = 100

Nruns = 5
CV = 5
InitRandomSeed = -1

######## BOOSTING STOP CRITERION  ########
sCBust = paC.Stop_Criterion("Nmax",[])
visual_DSNBoost = [1]
Inyection = "WholeO" # WholeO   PrevZo
Enphasis = "NeoBoosting"   #  RealAdaBoost NeoBoosting
alpha = 0.5
beta = 0.5
################################################################
#################### NEURAL NETWORM PARAMETERS ##################
#################################################################

nH = 4
    
fh_name = "tanh"
fo_name = "linear"
errFunc = "MSE"
visual = 0

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

######## CLUSTER PARAMETERS !!!!! ########
""" If we execute in da cluuster (spanish u) we overwritte the parameters """

N_epochs = 100
Learning_Rate = 0.01
BatchSize = 10

if ((Cluster_exec == 1)|(Console_exec == 1)):
    
    """ LIST OF POSSIBLE VALUES"""
    # Only 2 paramters !! The set of parameters to use and the repetition for the cluster.

    import pickle_lib as pkl
    param_list = pkl.load_pickle(output_folder + "BEST_params",1)

    params = param_list[int(param[0])]
    Repetition = int(param[1])  # Number of the repetitions in the cluster
    
    ### SET THE VALUES ###
    nH = params[0]
    N_epochs = params[1]
    fo_name = params[2]
    
    Learning_Rate = params[3]
    BatchSize =  params[4]
    
    Inyection =  params[5]
    Enphasis = params[6]
    
    alpha = params[7]
    beta = params[8]
    
    CV = 1
    Nruns = 1

#    Learning_Rate = Learning_Rate_list[int(param[2])]
    
    print "Number of hidden neurons: " + str(nH) 
    print "Number of Epochs: " + str(N_epochs) 
    print "Output function: " + str(fo_name) 
    print "Init Learning Rate: " + str(Learning_Rate) 
    print "Batch Size: " + str(BatchSize) 
    print "Inyection Type: " + str(Inyection) 
    print "Enphasis Type: " + str(Enphasis) 
    print "Alpha: " + str(alpha) 
    print "Beta: " + str(beta) 
    print "K fold: " + str(CV) 
    print "Number of Runs: " + str(Nruns) 


    
if (BP_F == 1):
    # Number of epochs, step and mommentum
    trainingAlg = paC.Training_Alg("BP",[100,0.001, 0.0])
    
if (BMBP_F == 1):
    # Number of epochs, step and number of partitions
    trainingAlg = paC.Training_Alg("BMBP",[N_epochs, Learning_Rate, BatchSize, 1])

if (ELM_F == 1):
    trainingAlg = paC.Training_Alg("ELM",["bias"])
    
if (ELMT_F == 1):
    trainingAlg = paC.Training_Alg("ELMT",[500, 0.0005, 20, "bias"])

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
########################## CAÑAAAA #########################
#################################################################


myDSNBoost = DSNBoost.CDSNBoost(nL = L,
                 CV = CV,
                 Nruns = Nruns,
                 Agregation = "RealAdaBoost", 
                 Enphasis = Enphasis,
                 Inyection = Inyection,
                 alpha = alpha, 
                 beta = beta,
                 InitRandomSeed = InitRandomSeed); # GentleBoost  RealAdaBoost

myDSNBoost.set_Train (Xtrain, Ytrain)
myDSNBoost.set_Val (Xtest, Ytest)
myDSNBoost.set_Test (Xtest, Ytest)

myDSNBoost.set_Base_Layer(mySLFN);

myDSNBoost.set_visual(visual_DSNBoost)
myDSNBoost.train();

myDSNBoost.output_stuff(output_folder,[int(param[0]), Repetition])          

print "Training Score: " + str(myDSNBoost.TrError)

print "Validation Score: " + str(myDSNBoost.ValError)

print "Test Score: " + str(myDSNBoost.TstError)

#print "Test Score: " + str(mySLFN.score(Xtest, Ytest))
