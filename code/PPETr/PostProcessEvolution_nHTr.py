# -*- coding: utf-8 -*-
"""
Created on Tue Mar 24 01:31:58 2015

@author: montoya
"""

import os
import numpy as np
import import_folders
import_folders.imp_folders(os.path.abspath(''))
import manutils as mu
import manugraphics as mg
import results_reader as rd
import matplotlib.pyplot as plt
import pickle_lib as pkl

plt.close("all")
""" THIS CODE READS THE INTERMEDIATE RESULTS OF THE LAYERS AND PLOTS THEM"""
""" FOR NOW ONLY ONE RUN PER FILE WITH NO CV !!!!! """ 


#################  PARAMETROS DEL BARRIDO ##################
## Here we introduce the init and end values of the cluster
## These are constant
fo_list = [0]
Learning_Rate_list = [0]
BatchSize_list = [3]
Inyection_list =  [0]
Enphasis_list = [1]
CV_List = [0]     # 1 is for training with all dataset
Nruns_List = [0]

### These are the ones we actually search
#nH_list = [8,10,12,14,16]
nH_list = [20,25,30,40]

nH_indx_list = [0,1,2,3]

Ninit_list = [1,4,8,16]
Ninit_indx_list = [0,1,2,3]

Roh_list = [1,2,4,8]
Roh_indx_list = [0,1,2,3]

N_epochs_list = [50,100,200]
N_epochs_indx_list = [0,1,2]

#alpha_list = [0.8,0.9,1]
#alpha_indx_list = [8,9,10]

alpha_list = [0.5,0.7,1]
alpha_indx_list = [5,7,10]

beta_list = [0.5]
beta_indx_list = [5]

## OTHER CONFIGURATION
#beta_indx_list = [0]
#alpha_indx_list = [0]

#################  GENERATE PARAMTERS SETS ##################

database_number = 2
beta_i = 0
alpha_i = 0

init_i = 1
roh_i = 0

Nepoch_i = 2

main_folder = "NeoEvoTrain"
#main_folder = "AdaEvo"

# Folder where we read the RAW files
folder = "../"+main_folder +"/ResultsNeoDSN"+str(database_number)
# Folder where we read and store the Preread files
base_folder_in = "../"+main_folder +"/PreRead/"   
# Folder where we store the graph
base_folder_out = "../"+main_folder +"/Gold/"    

results = pkl.load_pickle(base_folder_in + str(database_number) + "/" + "data_" + 
                            str(beta_indx_list[beta_i]) + "_"+ str(alpha_indx_list[alpha_i])+"_EVO",1)  # If the result file existis coz it was previously read

if (results == []):
    print "FILE NOT PREREAD"
    exit(0)

All_Object_list = results

All_Object_list = All_Object_list[init_i][roh_i][Nepoch_i]


def PostProcessEvolution_nH(All_Object_list):

    N_neurons = len(All_Object_list)
    
    nL_max = 100  # Maximum number of layers
    nLs_list = np.array(range(nL_max)) + 1
    
    # Obtain the score of training and validation for every layer and realization
    All_scoreTr_layers = []
    All_scoreVal_layers = []
    for i in range(N_neurons):
        scoreTr_layers, scoreVal_layers = rd.get_scores_layers (All_Object_list[i])
        All_scoreTr_layers.append(scoreTr_layers)
        All_scoreVal_layers.append(scoreVal_layers)
        
    # Get the average value and std for all the layers of the validation score
        
    All_aves_rea_val = []
    All_stds_rea_val = []
    
    for i in range(N_neurons):
        matrix = mu.convert_to_matrix(All_scoreVal_layers[i],nL_max)
        aves_rea_val, stds_rea_val = mu.get_ave_std_unfilled_matrix(matrix)
    
        All_aves_rea_val.append(aves_rea_val)
        All_stds_rea_val.append(stds_rea_val)
        
    # Get the average value and std for all the layers of the training score
    All_aves_rea_tr = []
    All_stds_rea_tr = []
    
    for i in range(N_neurons):
        matrix = mu.convert_to_matrix(All_scoreTr_layers[i],nL_max)
        aves_rea_tr, stds_rea_tr = mu.get_ave_std_unfilled_matrix(matrix)
    
        All_aves_rea_tr.append(aves_rea_tr)
        All_stds_rea_tr.append(stds_rea_tr)
        
    
    # Get the gammas values and their average and std
    All_gammas = []
    N_realizations = []
    
    All_aves_gammas = []
    All_stds_gam = []
    
    for i in range(N_neurons):
        
        gammas = rd.get_gammas (All_Object_list[i])    
        All_gammas.append(gammas)
        N_realizations.append(len(gammas))
        
        matrix = mu.convert_to_matrix(gammas)
        aves_gam, stds_gam = mu.get_ave_std_unfilled_matrix(matrix)
    
        All_aves_gammas.append(aves_gam)
        All_stds_gam.append(stds_gam)
        
    
    """ 1rd GRAPH """ 
    # Plot the Average Training and Validation score !!
    
    # Plot the average and shit
    mg.plot_acc_nL_nH(nH_list, All_aves_rea_val)
    
    plt.savefig(base_folder_out + str(database_number) +"/" + "Ave_Accu_val(nL,nH)_"
                + str(N_epochs_list[Nepoch_i])+".png")
                
    plt.close("all")
    mg.plot_acc_nL_nH(nH_list, All_aves_rea_tr)
    
    plt.savefig(base_folder_out + str(database_number) +"/" + "Ave_Accu_tr(nL,nH)_"
                + str(N_epochs_list[Nepoch_i])+".png")
    plt.close("all")
    
    
    """ 2rd GRAPH """ 
    # Plot the gammas evolution
    
    
    # Obtain the average Acc and nL for the different neurons applying
    # a common strop criteria.
    
    ave_val = np.ones((N_neurons,1))
    std_val = np.ones((N_neurons,1))
    
    ave_NLs = np.ones((N_neurons,1))
    std_Nls = np.ones((N_neurons,1))
    
    for nh_i in range(N_neurons):
    
        # Obtain nLs
        nLs = np.ones(N_realizations[nh_i])
        accuracies = np.ones((N_realizations[nh_i],1))
        
        for j in range(N_realizations[nh_i]):  # For every realization
            nLs[j] = mu.check_stop(All_gammas[nh_i][j],5 )
            
        # Get the NLs statistics
        ave_NLs[nh_i] = np.mean(nLs)
        std_Nls[nh_i] = np.std(nLs)
        
        for j in range (N_realizations[nh_i]):  # For every realization
            pene = All_scoreVal_layers[nh_i][j]
            accuracies[j] = pene[nLs[j]-1]
            
        ave_val[nh_i] = np.mean(accuracies)
        std_val[nh_i] = np.std(accuracies)
        
    mg.plot_accu_nH(nH_list,ave_val,std_val)
    
    plt.savefig(base_folder_out + str(database_number) +"/" + "Ave_Accu(nH)"
                + str(N_epochs_list[Nepoch_i])+".png")
                
    plt.close("all")
    
    mg.plot_nL_nH(nH_list, ave_NLs, std_Nls )
    plt.savefig(base_folder_out + str(database_number) +"/" + "Ave_nLs(nH)"
                + str(N_epochs_list[Nepoch_i])+".png")
    plt.close("all")
    
    """ 2rd GRAPH """ 
    # Plot the Average Training and Validation score !!
    
    mg.plot_3D_nH_nL(nH_list, nLs_list, All_aves_rea_tr)
    plt.savefig(base_folder_out + str(database_number) +"/" + "3D_Accu_tr(nH,nL)"
                + str(N_epochs_list[Nepoch_i])+".png")
                
    plt.close("all")
    
    mg.plot_3D_nH_nL(nH_list, nLs_list, All_aves_rea_val)
    plt.savefig(base_folder_out + str(database_number) +"/" + "3D_Accu_val(nH,nL)"
                + str(N_epochs_list[Nepoch_i])+".png")
                
    plt.close("all")
    
PostProcessEvolution_nH(All_Object_list)