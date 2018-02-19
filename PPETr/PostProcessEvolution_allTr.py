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
def PPE_allTr (PPE_p, main_folder, database_number, N_BEST = 30):

    Nb = len(PPE_p.beta_indx_list)
    Na = len(PPE_p.alpha_indx_list)
    
    Ni = len(PPE_p.Ninit_indx_list)
    Nr = len(PPE_p.Roh_indx_list)
    
    Ne = len(PPE_p.N_epochs_indx_list)
    Nh = len(PPE_p.nH_indx_list)
    
    dim_list = [Nb, Na, Ni, Nr, Ne, Nh]
    print dim_list
    #################  GENERATE PARAMTERS SETS ##################

    # Folder where we read the RAW files
    folder = "../"+main_folder +"/ResultsNeoDSN"+str(database_number) 
    # Folder where we read and store the Preread files
    base_folder_in = "../"+main_folder +"/PreReadResults/"   
    # Folder where we store the graph
    base_folder_out = "../"+main_folder +"/Gold/"+str(database_number) + "/all/"
  
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
    mu.create_dirs(base_folder_out)

    All_ave_val = np.array(All_ave_val)
    OMN_ordered, OMN_order = mu.sort_and_get_order(All_ave_val.flatten())
    
    text_file = open(base_folder_out +"BEST_OMN.txt", "w")
    
    BEST_indexes = []
    for i in range (N_BEST):
        index = OMN_order[-(i+1)]
        output_list = mu.get_all_indx_gen(index,dim_list)
        print output_list
        BEST_indexes.append(output_list)
        text_file.write("[beta: " + str(PPE_p.beta_list[PPE_p.beta_indx_list[output_list[0]]]) +
                        " alpha: " + str(PPE_p.alpha_list[PPE_p.alpha_indx_list[output_list[1]]]) +
                        " Ninit: " + str(PPE_p.Ninit_list[PPE_p.Ninit_indx_list[output_list[2]]]) +
                        " Roh: " + str(PPE_p.Roh_list[PPE_p.Roh_indx_list[output_list[3]]]) +
                        " Nep: " + str(PPE_p.N_epochs_list[PPE_p.N_epochs_indx_list[output_list[4]]]) +
                        " Nh: " + str(PPE_p.nH_list[PPE_p.nH_indx_list[output_list[5]]]) +
                        "] ")
        
        text_file.write("Tr: " + str( 1 - All_ave_tr[output_list[0]][output_list[1]][output_list[2]][output_list[3]][output_list[4]][output_list[5]]) + 
        " Tst: " + str( 1 - All_ave_val[output_list[0]][output_list[1]][output_list[2]][output_list[3]][output_list[4]][output_list[5]]) + 
        " std Tst: " + str(All_std_val[output_list[0]][output_list[1]][output_list[2]][output_list[3]][output_list[4]][output_list[5]]) + "\n" )
        
    text_file.close()
    
    return BEST_indexes


