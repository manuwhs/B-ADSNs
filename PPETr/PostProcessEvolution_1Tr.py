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



## OTHER CONFIGURATION
#beta_indx_list = [0]
#alpha_indx_list = [0]

#################  GENERATE PARAMTERS SETS ##################

#database_number = 3
#beta_i = 0
#alpha_i = 10
#Nepoch_i = 0
#nH_i = 19

def PPE_1Tr(PPE_p, main_folder, database_number,beta_i,alpha_i,Ninit_i, Roh_i,Nepoch_i,nH_i, nL_max = 200):  # Maximum number of layers):

    # Folder where we read the RAW files
    folder = "../"+main_folder +"/ResultsNeoDSN"+str(database_number)
    # Folder where we read and store the Preread files
    base_folder_in = "../"+main_folder +"/PreRead/"   
    # Folder where we store the graph
    base_folder_out = "../"+main_folder +"/Gold/" + str(database_number) +"/Nl/"  
    
    results = pkl.load_pickle(base_folder_in + str(database_number) + "/" + "data_" + 
                                str(beta_i) + "_"+ str(alpha_i)+"_EVO",1)  # If the result file existis coz it was previously read
    
    if (results == []):
        print "FILE NOT PREREAD"
        raise ValueError
        exit(0)
    
    All_Object_list = results
    nLs_list = np.array(range(nL_max)) + 1
    
    Object_list = All_Object_list[Ninit_i][Roh_i][Nepoch_i][nH_i]

    # Obtain the score of training and validation for every layer and realization
    scoreTr_layers, scoreVal_layers = rd.get_scores_layers (Object_list)
    
    # Get the average value and std for all the layers of the validation score
    matrix = mu.convert_to_matrix(scoreVal_layers,nL_max)
    aves_rea_val, stds_rea_val = mu.get_ave_std_unfilled_matrix(matrix)
    
    # Get the average value and std for all the layers of the training score
    matrix = mu.convert_to_matrix(scoreTr_layers,nL_max)
    aves_rea_tr, stds_rea_tr = mu.get_ave_std_unfilled_matrix(matrix)
    
    
    # Get the gammas values and their average and std
    gammas = rd.get_gammas (Object_list)     
    N_realizations = len(gammas)            # Number of realizations
    
    matrix = mu.convert_to_matrix(gammas)
    aves_gam, stds_gam = mu.get_ave_std_unfilled_matrix(matrix)
    

    base_folder_out = base_folder_out + "a:"+ str(alpha_i)+"/" +"b:" +str(beta_i)+ "/"
    base_folder_out = base_folder_out + "Ninit:"+ str(Ninit_i)+"/" +"Roh:" +str(Roh_i)+ "/"
    mu.create_dirs(base_folder_out)
    
    """ 1st GRAPH """
    # Plot all realizations for the given number of Epoch and Neurons
    mg.plot_all_realizations_EVO (scoreTr_layers,scoreVal_layers )
    
    # Save figure !!
    plt.savefig(base_folder_out + "All_rea_"
                + str(PPE_p.N_epochs_list[PPE_p.N_epochs_indx_list[Nepoch_i]]) +"_" + str(PPE_p.nH_list[PPE_p.nH_indx_list[nH_i]]) +".png")
    plt.close("all")
    
    """ 2nd GRAPH """
    # Plot the Average Training and Validation score !!
    # Plot the average and shit
    mg.plot_tr_val_nL(nLs_list, aves_rea_val, aves_rea_tr,stds_rea_val,stds_rea_tr)
    plt.savefig(base_folder_out + "Ave_Acc(nL)"
                + str(PPE_p.N_epochs_list[PPE_p.N_epochs_indx_list[Nepoch_i]]) +"_" + str(PPE_p.nH_list[PPE_p.nH_indx_list[nH_i]]) +".png")
    
    plt.close("all")
    
    """ 3rd GRAPH """ 
    # Plot the average gamma value in function of the number of layers 
    # Also plot where the nL would be stopped applying the rule.
    
    nLs = np.ones(N_realizations)
    for i in range(N_realizations):  # For every realization
        nLs[i] = mu.check_stop(gammas[i])
        
    mg.plot_gamma_nL(aves_gam,stds_gam)
    
    plt.scatter(nLs, np.mean(aves_gam) * np.ones(N_realizations))
    plt.savefig(base_folder_out + "Ave_gamma(nLs)"
                + str(PPE_p.N_epochs_list[PPE_p.N_epochs_indx_list[Nepoch_i]]) +"_" + str(PPE_p.nH_list[PPE_p.nH_indx_list[nH_i]]) +".png")
    
    plt.close("all")
    
    """ 4th GRAPH """
    # Plot the average Accuracy and Number of layers depending on the
    # stopping condition given by the ngamma
    
    ngammas_list = range(1,20)
    
    accuracies = np.ones((N_realizations,1))
    ave_val = np.ones((len(ngammas_list),1))
    std_val = np.ones((len(ngammas_list),1))
    
    ave_NLs = np.ones((len(ngammas_list),1))
    std_Nls = np.ones((len(ngammas_list),1))
    
    for i in range (len(ngammas_list)):
        # Obtain nLs
        nLs = np.ones(N_realizations)
        
        for j in range(N_realizations):  # For every realization
            nLs[j] = mu.check_stop(gammas[j],ngammas_list[i] )
            
        # Get the NLs statistics
        ave_NLs[i] = np.mean(nLs)
        std_Nls[i] = np.std(nLs)
        
        for j in range (N_realizations):  # For every realization
            accuracies[j] = scoreVal_layers[j][nLs[j]-1]
            
        ave_val[i] = np.mean(accuracies)
        std_val[i] = np.std(accuracies)
        
    mg.plot_accu_ngamma(ngammas_list,ave_val,std_val )
    plt.savefig(base_folder_out + "/" + "Ave_Accu(ngamma)_"
                + str(PPE_p.N_epochs_list[PPE_p.N_epochs_indx_list[Nepoch_i]]) +"_" + str(PPE_p.nH_list[PPE_p.nH_indx_list[nH_i]]) +".png")
    plt.close("all")
    
    mg.plot_nL_ngamma(ngammas_list, ave_NLs, std_Nls )
    plt.savefig(base_folder_out + "Ave_nL(ngamma)_"
                + str(PPE_p.N_epochs_list[PPE_p.N_epochs_indx_list[Nepoch_i]]) +"_" + str(PPE_p.nH_list[PPE_p.nH_indx_list[nH_i]]) +".png")
    plt.close("all")


#PPE_1(database_number,beta_i,alpha_i,Nepoch_i,nH_i)


