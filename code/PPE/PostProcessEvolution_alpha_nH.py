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
""" THIS CODE AIMS TO READ THE DATA FOR ALL [beta][alpha][Nepoch][Nh],
Apply a stop condition for NL and obtain the ave and std of every one """


def PPE_alpha_nH (PPE_p, main_folder, database_number, beta_i, Nepoch_i):

    Nb = len(PPE_p.beta_indx_list)
    Na = len(PPE_p.alpha_indx_list)
    Ne = len(PPE_p.N_epochs_indx_list)
    Nh = len(PPE_p.nH_indx_list)

    #main_folder = "AdaEvo"
    # Folder where we read the RAW files
    folder = "../"+main_folder +"/ResultsNeoDSN"+str(database_number)
    # Folder where we read and store the Preread files
    base_folder_in = "../"+main_folder +"/PreReadResults/"   
    # Folder where we store the graph
    base_folder_out = "../"+main_folder +"/Gold/"+str(database_number) + "/alpha_Nh/"    

    
    # WE read the needed files
    
    All_ave_tr = []
    All_std_tr = []
    All_ave_val = []
    All_std_val = []
    
    for alpha_i in range(len(PPE_p.alpha_indx_list)):
                    
        ## First check if the file with the read data already exists
        results = pkl.load_pickle(base_folder_in + str(database_number) + "/" + "data_" + 
                  str(PPE_p.beta_indx_list[beta_i]) + "_"+ str(PPE_p.alpha_indx_list[alpha_i])+"_EVO",1)  # If the result file existis coz it was previously read
    
        All_ave_tr.append(results[0])
        All_std_tr.append(results[1])
        All_ave_val.append(results[2])
        All_std_val.append(results[3])
            
    
    # Now we have:
    
    #All_std_tr[beta][alpha][Nepoch][Nh]
    #All_ave_val = np.array(All_ave_val)
    
    
    """ Alpha and Nh for a given Nh """
    
    Ave_val_alNh = []
    Ave_tr_alNh = []
    
    aux = 0
    for alpha_i in range(Na):
        Ave_val_alNh.append([])
        Ave_tr_alNh.append([])
        for nH_i in range(Nh):
            pene = All_ave_val[alpha_i][Nepoch_i][nH_i]
            Ave_val_alNh[aux].append(pene)   
            pene = All_ave_tr[alpha_i][Nepoch_i][nH_i]
            Ave_tr_alNh[aux].append(pene) 
        aux += 1
    
    Ave_val_alNh = np.array(Ave_val_alNh)
    Ave_tr_alNh = np.array(Ave_tr_alNh)
    
    """ 2rd GRAPH """ 
    # Plot the Average Training and Validation as a function of alpha and nH !!
#    base_folder_out = base_folder_out   # + "b: "+ str(beta_i) + "/"
    
    base_folder_out = base_folder_out + "epoch: "+ str(PPE_p.N_epochs_list[PPE_p.N_epochs_indx_list[Nepoch_i]]) + "/"
    
    mu.create_dirs(base_folder_out)
    
    mg.plot_3D_a_nH(PPE_p.alpha_list[PPE_p.alpha_indx_list], PPE_p.nH_list[PPE_p.nH_indx_list], Ave_tr_alNh)
    plt.savefig(base_folder_out + "3D_accu_tr(alpha,nH)_b:" + str(PPE_p.beta_list[PPE_p.beta_indx_list[beta_i]]) + " "
                + str(PPE_p.N_epochs_list[PPE_p.N_epochs_indx_list[Nepoch_i]])+".png")
    plt.close("all")
    
    mg.plot_3D_a_nH(PPE_p.alpha_list[PPE_p.alpha_indx_list], PPE_p.nH_list[PPE_p.nH_indx_list], Ave_val_alNh)
    plt.savefig(base_folder_out + "/" + "3D_accu_val(alpha,nH)_b:" + str(PPE_p.beta_list[PPE_p.beta_indx_list[beta_i]]) + " "
                + str(PPE_p.N_epochs_list[PPE_p.N_epochs_indx_list[Nepoch_i]])+".png")
    plt.close("all")      


