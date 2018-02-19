# -*- coding: utf-8 -*-
"""
Created on Tue Mar 24 01:31:58 2015

@author: montoya
"""

#################################################
############ HEADING FOR WORKING ################
#################################################

Cluster_exec = 1
Spyder_exec = 0
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
    output_folder = "/export/clusterdata/mmontoya/ResultsDSN3shit/"
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
import DSN
import SLFN 
import paramClasses as paC

plt.close('all')


#################################################
################# LOAD DATASET ##################
#################################################

#mat = scipy.io.loadmat(dataset_folder +'abalone.mat')
mat = scipy.io.loadmat(dataset_folder +'image.mat')
#mat = scipy.io.loadmat(dataset_folder +'kwok.mat')
#mat = scipy.io.loadmat(dataset_folder +'waveform.mat')

#mat = scipy.io.loadmat(dataset_folder +'breast.mat')
#mat = scipy.io.loadmat(dataset_folder +'hepatitis.mat')
#mat = scipy.io.loadmat(dataset_folder +'ripley.mat')

#mat = scipy.io.loadmat(dataset_folder +'ionosfera.mat')
#mat = scipy.io.loadmat(dataset_folder +'crabs.mat')

Xtrain = mat["X_tr"]
Ytrain = mat["T_tr"]

Xtest = mat["X_tst"]
Ytest = mat["T_tst"]

# 10 - 15 layers, 6 neurons, [500, 0.0005, 10]
# kwok, 13 neurons 100,
#################################################################
#################### DATA PREPROCESSING #########################
#################################################################
    
#%% Normalize data
scaler = preprocessing.StandardScaler().fit(Xtrain)
Xtrain = scaler.transform(Xtrain)            
Xtest = scaler.transform(Xtest)       
        
#################################################################
########################## DSN PARAMETERS ########################
#################################################################
L = 100
    
Nruns =  1 # These are 1 for Boosting !!
CV = 1
InitRandomSeed = -1

######## BOOSTING STOP CRITERION  ########
sCBust = paC.Stop_Criterion("Nmax",[])
visual_DSN = DSN.DSN_visual(verbose = 1,
                                      store_layers_scores = 1,
                                      store_layers_soft_error = 0,
                                      plot_results_layers = 0
                                      )        
nP = 10
################################################################
#################### NEURAL NETWORM PARAMETERS ##################
#################################################################

nH = 20
    
fh_name = "tanh"
fo_name = "tanh"
errFunc = "MSE"
visual = 0

######## WEIGHT INITIALIZATION  ########
#initDistrib = paC.Init_Distrib("default", ["uniform",-1,1])
#initDistrib = paC.Init_Distrib("default", ["normal",0,1])
initDistrib = paC.Init_Distrib("deepLearning", ["DL1"])

######## STOP CRITERION  ########
stopCriterion = paC.Stop_Criterion("Nmax",[])

########  REGULARIZATION   ########
regularization = paC.Regularization("NoL2",[0.000005])
######## TRAINING ALGORITHM  ########
# Parameters of the algorithms 
N_epochs = 200
Learning_Rate = 0.01
BatchSize = 2

######## CLUSTER PARAMETERS !!!!! ########
""" If we execute in da cluuster (spanish u) we overwritte the parameters """

if ((Cluster_exec == 1)|(Console_exec == 1)):
    
    """ LIST OF POSSIBLE VALUES"""
    nH_list = [2,3,4,5,6,7,8,9,10,11,
               12,13,14,15,16,17,18,19,20,21,
               22,23,24,25,26,27,28,29,30,31,
               32,33,34,35,36,37,38,39,40,41,
               42,43,44,45,46,47,48,49,50,
               53,56,59,62,65,68,71,74,77,80,
               83,86,89,92,95,98,101,104,107,110]
               
    N_epochs_list = [50,100,200,300,400]
    fo_list = ["linear","tanh"]
    
    Learning_Rate_list = [0.01,0.005,0.002,0.001]
    BatchSize_list = [1,2,3,4,5,10,20]

    CV_List = [1,5,10]     # 1 is for training with all dataset
    Nruns_List = [1,2,3,4,5,6,7,8,9,10]
    
    ### SET THE VALUES ###
    nH = nH_list[int(param[0])]
    N_epochs = N_epochs_list[int(param[1])]
    fo_name = fo_list[int(param[2])]
    
    Learning_Rate = Learning_Rate_list[int(param[3])]
    BatchSize =  BatchSize_list[int(param[4])]
    
    CV = CV_List [int(param[5])]
    Nruns = Nruns_List [int(param[6])]

    Repetition = int(param[7])  # Number of the repetitions in the cluster
    
#    Learning_Rate = Learning_Rate_list[int(param[2])]
    
    print "Number of hidden neurons: " + str(nH) 
    print "Number of Epochs: " + str(N_epochs) 
    print "Output function: " + str(fo_name) 
    print "Init Learning Rate: " + str(Learning_Rate) 
    print "Batch Size: " + str(BatchSize) 
    print "K fold: " + str(CV) 
    print "Number of Runs: " + str(Nruns) 
    
    if (Cluster_exec == 1):  # To make sure we dont show results in the cluster.
        visual_DSN.plot_results_layers = 0
    if ((CV == 1)&(Nruns == 1)):
        visual_DSN.store_layers_scores = 1
        visual_DSN.store_layers_soft_error = 1
    else:
        visual_DSN.store_layers_scores = 0
        visual_DSN.store_layers_soft_error = 0
        
######## TRAINING ALGORITHM  ########

BP_F = 0
BMBP_F = 0

if (BatchSize == 1):
    BP_F = 1
else:
    BMBP_F = 1

ELM_F = 0
ELMT_F = 0

LDA_F = 0
LDAT_F = 0

if (BP_F == 1):
    # Number of epochs, step and mommentum
    momentum = 0
    trainingAlg = paC.Training_Alg("BP",[N_epochs,Learning_Rate, momentum])
    
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
########################## DSN Using #########################
#################################################################

myDSN = DSN.CDSN(nL = L,
                 nP = 10,
                 CV = CV,
                 Nruns = Nruns,
                 InitRandomSeed = InitRandomSeed); # GentleBoost  RealAdaBoost

myDSN.set_Train (Xtrain, Ytrain)
myDSN.set_Val (Xtest, Ytest)
myDSN.set_Test (Xtest, Ytest)

myDSN.set_Base_Layer(mySLFN);

myDSN.set_visual(visual_DSN)
myDSN.train();


myDSN.output_stuff(output_folder,[int(param[0]),int(param[1]),int(param[2]),
                                       int(param[3]), int(param[4]),int(param[5]),
                                       int(param[6]), Repetition])          

print "Training Score: " + str(myDSN.TrError)

print "Validation Score: " + str(myDSN.ValError)

print "Test Score: " + str(myDSN.TstError)

#print "Test Score: " + str(mySLFN.score(Xtest, Ytest))


