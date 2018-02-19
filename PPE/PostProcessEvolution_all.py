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

def PPE_all (PPE_p, main_folder, database_number, N_BEST = 30):
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
    
    All_ave_val = np.array(All_ave_val)
#    print "PENE"
#    print All_ave_val.shape
    OMN_ordered, OMN_order = mu.sort_and_get_order(All_ave_val.flatten())
    
    mu.create_dirs(base_folder_out)
    text_file = open(base_folder_out +"BEST_OMN.txt", "w")
    
    
    BEST_indexes = []
    for i in range (N_BEST):
        index = OMN_order[-(i+1)]
        n_b, n_a, n_e, n_h = mu.get_all_indx(index,Nb,Na,Ne,Nh)
        BEST_indexes.append([n_b, n_a, n_e, n_h])
#        print n_b, n_a, n_e, n_h
#        print "[beta: " + str(PPE_p.beta_list[n_b])
#        print " alpha: " + str(PPE_p.alpha_list[n_a])
#        print " Nep: " + str(PPE_p.N_epochs_list[n_e])
#        print " Nh: " + str(PPE_p.nH_list[n_h])
        
        text_file.write("[beta: " + str(PPE_p.beta_list[PPE_p.beta_indx_list[n_b]]) +
                        " alpha: " + str(PPE_p.alpha_list[PPE_p.alpha_indx_list[n_a]]) +
                        " Nep: " + str(PPE_p.N_epochs_list[PPE_p.N_epochs_indx_list[n_e]]) +
                        " Nh: " + str(PPE_p.nH_list[PPE_p.nH_indx_list[n_h]]) +
                        "] ")
        
        text_file.write("Tr: " + str( 1 - All_ave_tr[n_b][n_a][n_e][n_h]) + 
        " Tst: " + str( 1 - All_ave_val[n_b][n_a][n_e][n_h]) + 
        " std Tst: " + str(All_std_val[n_b][n_a][n_e][n_h]) + "\n" )
        
    text_file.close()

    return BEST_indexes


#PPE_all(database_number)