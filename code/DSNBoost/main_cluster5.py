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
    import manutils as mu
    print param

    database_num = int(os.path.basename(__file__).split(".")[-2][-1])
    # Configure cluster folders for data input and output
    output_folder = "/export/clusterdata/mmontoya/" + str(database_num) + "/"
    dataset_folder = "/export/clusterdata/mmontoya/data/"
    
elif (Spyder_exec == 1): # If it is executed from spyder
    import os
    import sys
    whole_path = os.path.abspath('')
    folder_name = os.path.basename(whole_path)
    root_path = whole_path.split(folder_name)[0]
    sys.path.append(root_path)   # Include the root path into the "import" paths

    import import_folders
    import_folders.imp_folders(root_path)
    import manutils as mu
    
    database_num = int(os.path.basename(__file__).split(".")[-2][-1])
    output_folder = "../Results/"+ str(database_num) + "/"
    dataset_folder = "../data/"

elif (Console_exec == 1): # If executed using command line from root folder.
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
    import manutils as mu
    database_num = int(os.path.basename(__file__).split(".")[-2][-1])
    print database_num
    print param

    # Configure cluster folders for data input and output

    output_folder = "../Results/"+ str(database_num) + "/"
    dataset_folder = "../data/"
    mu.create_dirs(output_folder)
    
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

if (database_num == 1):
    mat = scipy.io.loadmat(dataset_folder +'abalone.mat')
elif (database_num == 2):
    mat = scipy.io.loadmat(dataset_folder +'kwok.mat')
elif (database_num == 3):
    mat = scipy.io.loadmat(dataset_folder +'image.mat')
elif (database_num == 4):
    mat = scipy.io.loadmat(dataset_folder +'waveform.mat')
elif (database_num == 5):
    mat = scipy.io.loadmat(dataset_folder +'ionosfera.mat')
elif (database_num == 6):
    mat = scipy.io.loadmat(dataset_folder +'ripley.mat')
elif (database_num == 7):
    mat = scipy.io.loadmat(dataset_folder +'hepatitis.mat')
#mat = scipy.io.loadmat(dataset_folder +'breast.mat')




#mat = scipy.io.loadmat(dataset_folder +'crabs.mat')

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
        
L = 300        # Maximum number of layers 

Nruns = 1       # Number of runs of this process
CV = 1          # K fold of the crossvalidation. If 1 -> Omniscient
InitRandomSeed = -1  # Random Seed to redo the same experiment.

sCBust = paC.Stop_Criterion("Nmax",[]) ## BOOSTING STOP CRITERION 
visual_DSNBoost = DSNBoost.DSN_visual(verbose = 0,
                                      store_layers_scores = 1,
                                      store_layers_soft_error = 0,
                                      plot_results_layers = 1
                                      )         
                                      
# Visualizing intermediate results




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

######## STOP CRITERION  ########
stopCriterion = paC.Stop_Criterion("Nmax",[])

########  REGULARIZATION   ########
regularization = paC.Regularization("L2",[0.000005])

######## TRAINING ALGORITHM  ########
# Parameters of the algorithms 
N_epochs = 100
Learning_Rate = 0.01
BatchSize = 1
Roh = 1
Ninit = 6

######## CLUSTER PARAMETERS !!!!! ########
""" If we execute in da cluuster (spanish u) we overwritte the parameters """

if ((Cluster_exec == 1)|(Console_exec == 1)):
    
    """ LIST OF POSSIBLE VALUES"""
    nH_list = [2,3,4,5,6,7,8,9,10,11,
               12,13,14,15,16,17,18,19,20,21,
               22,23,24,25,26,27,28,29,30,31,
               32,33,34,35,36,37,38,39,40,41,
               42,43,44,45,46,47,48,49,50]
               
    N_epochs_list = [25,50,100,150,200,300]
    fo_list = ["linear","tanh"]
    
    Ninit_list = [1,4,8,16]
    Roh_list = [1,2,4,8]
    
    Inyection_list =  ["WholeO", "PrevZo", "NoInyection"]
    Enphasis_list = [ "RealAdaBoost","NeoBoosting", "RA-we","ManuBoost","ManuBoost2","NeoBoosting2"] 
    
    alpha_list = [0,0.1,0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1]
    beta_list = [0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1]
    
    CV_List = [1,5,10,20,50]     # 1 is for training with all dataset
    Nruns_List = [1,2,3,4,5,6,7,8,9,10]
    
    ### SET THE VALUES ###
    nH = nH_list[int(param[0])]
    N_epochs = N_epochs_list[int(param[1])]
    fo_name = fo_list[int(param[2])]
    
    Ninit = Ninit_list[int(param[3])]
    Roh =  Roh_list[int(param[4])]
    
    Inyection =  Inyection_list[int(param[5])]
    Enphasis = Enphasis_list[int(param[6])]
    
    alpha = alpha_list [int(param[7])]
    beta = beta_list [int(param[8])]
    
    CV = CV_List [int(param[9])]
    Nruns = Nruns_List [int(param[10])]

    Repetition = int(param[11])  # Number of the repetitions in the cluster
    
#    Learning_Rate = Learning_Rate_list[int(param[2])]
    
    print "Number of hidden neurons: " + str(nH) 
    print "Number of Epochs: " + str(N_epochs) 
    print "Output function: " + str(fo_name) 
    print "Init Learning Rate: " + str(Learning_Rate) 
    print "Batch Size: " + str(BatchSize) 
    print "Ninit: " + str(Ninit) 
    print "Roh: " + str(Roh) 
    print "Inyection Type: " + str(Inyection) 
    print "Enphasis Type: " + str(Enphasis) 
    print "Alpha: " + str(alpha) 
    print "Beta: " + str(beta) 
    print "K fold: " + str(CV) 
    print "Number of Runs: " + str(Nruns) 
    
    if (Cluster_exec == 1):  # To make sure we dont show results in the cluster.
        visual_DSNBoost.plot_results_layers = 0
        
    if ((CV == 1)&(Nruns == 1)):
        visual_DSNBoost.store_layers_scores = 1
        visual_DSNBoost.store_layers_soft_error = 1
    else:
        visual_DSNBoost.store_layers_scores = 0
        visual_DSNBoost.store_layers_soft_error = 0

######## WEIGHT INITIALIZATION  ########
#initDistrib = paC.Init_Distrib("default", ["uniform",-0.01,0.01])
#initDistrib = paC.Init_Distrib("default", ["normal",0,1])
initDistrib = paC.Init_Distrib("deepLearning", ["DL1", Ninit])

######## TRAINING ALGORITHM  ########

BP_F = 0
BMBP_F = 0

if (BatchSize == 1):
    BP_F = 1
#    print "pene"
else:
    BMBP_F = 1

ELM_F = 0
ELMT_F = 0

LDA_F = 0
LDAT_F = 0

if (BP_F == 1):
    # Number of epochs, step and mommentum
    trainingAlg = paC.Training_Alg("BP",[N_epochs,Learning_Rate, Roh, 0.0])
    
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
                 Agregation = "RealAdaBoost",   # GentleBoost  RealAdaBoost
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

myDSNBoost.output_stuff(output_folder,[int(param[0]),int(param[1]),int(param[2]),
                                       int(param[3]), int(param[4]),int(param[5]),
                                       int(param[6]),int(param[7]),int(param[8]),
                                       int(param[9]),int(param[10]), Repetition])          

print "Training Score: " + str(myDSNBoost.TrError)

print "Validation Score: " + str(myDSNBoost.ValError)

print "Test Score: " + str(myDSNBoost.TstError)

#print "Test Score: " + str(mySLFN.score(Xtest, Ytest))


# Usar Validacion para ver lo que pasa y variar alpha en funcion de ello.
# Si Tr sube y val baja, disminuir alpha
# Si Tr sube y val sube, mantenerse
# Si Tr baja y val baja -> Underfitting, bajar alpha
# Si val sube y Train baja -> Mantenerse.
