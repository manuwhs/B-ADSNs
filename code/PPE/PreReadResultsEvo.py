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
import results_reader as rd
import matplotlib.pyplot as plt
import pickle_lib as pkl

plt.close("all")
""" THIS CODE AIMS TO READ THE DATA FOR ALL [beta][alpha][Nepoch][Nh],
Apply a stop condition for NL and obtain the ave and std of every one """


def PPE_readResults(PPE_p, main_folder,database_number):

    # Folder where we read and store the Preread files
    base_folder_in = "../"+main_folder +"/PreRead/"   
    # Folder where we store the graph
    base_folder_out = "../"+main_folder +"/PreReadResults/"   
    
    # For evety combination of fucking useless parameters:
    
    Total_nepochs = len(PPE_p.N_epochs_indx_list)
    Total_nH = len(PPE_p.nH_indx_list)
    
    ave_val = np.ones((Total_nepochs,Total_nH))
    ave_tr = np.ones((Total_nepochs,Total_nH))
        
    std_val = np.ones((Total_nepochs,Total_nH))
    std_tr = np.ones((Total_nepochs,Total_nH))
    
    mu.create_dirs(base_folder_out + str(database_number) + "/")
    
    for beta_i in range(len(PPE_p.beta_indx_list)):
        for alpha_i in range(len(PPE_p.alpha_indx_list)):
                        
            ## First check if the file with the read data already exists
            albe_objects = pkl.load_pickle(base_folder_in + str(database_number) + "/" + "data_" + 
                      str(PPE_p.beta_indx_list[beta_i]) + "_"+ str(PPE_p.alpha_indx_list[alpha_i])+"_EVO",1)  # If the result file existis coz it was previously read
    
            
            
            for Nepoch_i in range(len(albe_objects)):
                for nH_i in range(len(albe_objects[Nepoch_i])):
#                    print len(albe_objects), len(albe_objects[Nepoch_i])
                    if (PPE_p.CV_List[0] != 0):  # If we are not loading evolution, but old style final results 
                        a,b,c,d = mu.get_final_results_CV(albe_objects[Nepoch_i][nH_i])
                    else:
                        a,b,c,d = mu.get_final_results_Evo(albe_objects[Nepoch_i][nH_i], ngammas = 5)
                    
                    ave_tr[Nepoch_i][nH_i] = a
                    std_tr[Nepoch_i][nH_i] = b
                    ave_val[Nepoch_i][nH_i] = c
                    std_val[Nepoch_i][nH_i]  = d
    
            rd.save_results(base_folder_out + str(database_number) + "/" + "data_"+ 
                                str(PPE_p.beta_indx_list[beta_i]) + "_"+ str(PPE_p.alpha_indx_list[alpha_i])+"_EVO", [ave_tr,std_tr,ave_val,std_val])
    
    """ NOW we have to:
        For every combination [beta][alpha][Nepoch][Nh]
            Apply the stop condition, calculate the nLs of every realization.
            Then calculate the average accuracy of tr and tst for every 

    WE should then get them into an array and get the maximum 
    
    MAKE A FILE THAT READS THEM AND DOES THIS  
    
    """


