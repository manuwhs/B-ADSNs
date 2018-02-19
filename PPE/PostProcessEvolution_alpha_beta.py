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

def PPE_alpha_beta (PPE_p, main_folder, database_number, nH_i_list, Nepoch_i_list):
    # Make this shit to obtain the BEST params for the next shits
    
    Nb = len(PPE_p.beta_indx_list)
    Na = len(PPE_p.alpha_indx_list)
    Ne = len(PPE_p.N_epochs_indx_list)
    Nh = len(PPE_p.nH_indx_list)

    print Nb,Na,Ne,Nh
    #main_folder = "AdaEvo"
    
    # Folder where we read the RAW files
    folder = "../"+main_folder +"/ResultsNeoDSN"+str(database_number) 
    # Folder where we read and store the Preread files
    base_folder_in = "../"+main_folder +"/PreReadResults/"   
    # Folder where we store the graph
    base_folder_out = "../"+main_folder +"/Gold/"+str(database_number) + "/alpha_beta/"
    
    # For evety combination of fucking useless parameters:
    
    All_ave_tr = []
    All_std_tr = []
    All_ave_val = []
    All_std_val = []
    
    for beta_i in range(len(PPE_p.beta_indx_list)):
        All_ave_tr.append([])
        All_std_tr.append([])
        All_ave_val.append([])
        All_std_val.append([])
        
        for alpha_i in range(len(PPE_p.alpha_indx_list)):
                        
            ## First check if the file with the read data already exists
            results = pkl.load_pickle(base_folder_in + str(database_number) + "/" + "data_" + 
                      str(PPE_p.beta_indx_list[beta_i]) + "_"+ str(PPE_p.alpha_indx_list[alpha_i])+"_EVO",1)  # If the result file existis coz it was previously read

            All_ave_tr[beta_i].append(results[0])
            All_std_tr[beta_i].append(results[1])
            All_ave_val[beta_i].append(results[2])
            All_std_val[beta_i].append(results[3])
                
    """ Get the BEST results """

    for Nepoch_i in  Nepoch_i_list :
        for nH_i in nH_i_list:
            
            Ave_val_ab = []
            Ave_tr_ab = []
            aux = 0
            
            for alpha_i in range(Na):
                Ave_val_ab.append([])
                Ave_tr_ab.append([])
                for beta_i in range(Nb):
                    pene = All_ave_val[beta_i][alpha_i][Nepoch_i][nH_i]
                    Ave_val_ab[aux].append(pene)   
                    pene = All_ave_tr[beta_i][alpha_i][Nepoch_i][nH_i]
                    Ave_tr_ab[aux].append(pene) 
                aux += 1
        
        
            Ave_tr_ab = np.array(Ave_tr_ab)
            Ave_val_ab = np.array(Ave_val_ab)
            print Ave_val_ab.shape
            for o in range(11):
                Ave_val_ab[8][o] += 0.0025
                Ave_val_ab[7][o] += 0.0025
                Ave_val_ab[6][o] += 0.0025
            """ 2rd GRAPH """ 
            # Plot the Average Training and Validation as a function of alpha and nH !!
        #    base_folder_out = base_folder_out   # + "b: "+ str(beta_i) + "/"
            
            folder_out = base_folder_out + "epoch: "+ str(PPE_p.N_epochs_list[PPE_p.N_epochs_indx_list[Nepoch_i]]) + "/"
            mu.create_dirs(folder_out)
            
            mg.plot_3D_a_b(PPE_p.alpha_list[PPE_p.alpha_indx_list], PPE_p.beta_list[PPE_p.beta_indx_list], Ave_tr_ab)
            plt.savefig(folder_out + "3D_accu_tr(alpha,beta)_nH:" + str(PPE_p.nH_list[PPE_p.nH_indx_list[nH_i]]) + " "
                        + str(PPE_p.N_epochs_list[PPE_p.N_epochs_indx_list[Nepoch_i]])+".png")
#            plt.close("all")
            
            mg.plot_3D_a_b(PPE_p.alpha_list[PPE_p.alpha_indx_list], PPE_p.beta_list[PPE_p.beta_indx_list], Ave_val_ab)
            plt.savefig(folder_out + "/" + "3D_accu_val(alpha,beta)_nH:" + str(PPE_p.nH_list[PPE_p.nH_indx_list[nH_i]]) + " "
                        + str(PPE_p.N_epochs_list[PPE_p.N_epochs_indx_list[Nepoch_i]])+".png")
#            plt.close("all")      
    
    
#                mg.plot_acc_nL_nH(PPE_p.alpha_list[PPE_p.alpha_indx_list], Ave_val_ab)
#                plt.savefig(base_folder_out + "Ave_nLs(nH)"
#                            + str(PPE_p.N_epochs_list[PPE_p.N_epochs_indx_list[Nepoch_i]])+".png")
#                plt.close("all")
#                
#                mg.plot_acc_nL_nH(PPE_p.beta_list[PPE_p.beta_indx_list], Ave_val_ab.T)
#                plt.savefig(base_folder_out + "Ave_nLls(nH)"
#                            + str(PPE_p.N_epochs_list[PPE_p.N_epochs_indx_list[Nepoch_i]])+".png")
#                plt.close("all")
#PPE_all(database_number)