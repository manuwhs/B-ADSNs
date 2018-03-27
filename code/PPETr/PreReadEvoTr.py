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
import pickle_lib as pkl

def PPE_readTr(PPE_p, main_folder,database_number, repetitions = 20): 
    """ PREREADING STRATEGY """

    # Folder where we read the RAW files
    folder = "../"+main_folder +"/ResultsNeoDSN"+str(database_number)
    # Folder where we read and store the Preread files
    base_folder_in = "../"+main_folder +"/PreRead/"   
    # Folder where we store the graph
    base_folder_out = "../"+main_folder +"/Gold/"   
    
    
    # We will build a file for every pair [beta, alpha]
    # Every file will contain the "N_realizations" for different values of [Nepoch, Nh]
    """ $$$$$$$$$$$$  Evolution Error  $$$$$$$$$$$$$$$$ """
    
    mu.create_dirs(base_folder_in + str(database_number) + "/")
    
    # For evety combination of fucking useless parameters:
    for beta_i in range(len(PPE_p.beta_indx_list)):
        for alpha_i in range(len(PPE_p.alpha_indx_list)):
                        
            ## First check if the file with the read data already exists
            results = pkl.load_pickle(base_folder_in + str(database_number) + "/" + "data_" + 
                      str(PPE_p.beta_indx_list[beta_i]) + "_"+ str(PPE_p.alpha_indx_list[alpha_i])+"_EVO",1)  # If the result file existis coz it was previously read
            
            read_flag = 1
            
            if ((results != [])&(read_flag == 0)):
                All_Object_list = results
                
            else:
                All_Object_list = []  # List that will contain of the object lists for a given [beta_alpha]
                                      # It will contain All_Object_list[Nepoch][Nh] = N_realizations objects
                
                # Get the list of parameters and load them into All_Object_list
                for n_init_i in range (len(PPE_p.Ninit_indx_list)):
                    All_Object_list.append([])
                    
                    for n_roh_i in range (len(PPE_p.Roh_indx_list)):    
                        All_Object_list[n_init_i].append([])
                        
                        for n_epoch_i in range(len(PPE_p.N_epochs_indx_list)): 
                            All_Object_list[n_init_i][n_roh_i].append([])
                            
                            
                            for nH_i in range(len(PPE_p.nH_indx_list)):
                                
                                Parameters = [PPE_p.nH_indx_list[nH_i],PPE_p.N_epochs_indx_list[n_epoch_i],
                                                   1,
                                                   PPE_p.Ninit_indx_list[n_init_i],
                                                   PPE_p.Roh_indx_list[n_roh_i],
                                                   PPE_p.Inyection_list[0],PPE_p.Enphasis_list[0],
                                                   PPE_p.alpha_indx_list[alpha_i],PPE_p.beta_indx_list[beta_i],
                                                   0,0]
                                                   
                                Object_list = rd.load_objects_repetitions(folder, [Parameters], repetitions)
                                All_Object_list[n_init_i][n_roh_i][n_epoch_i].append(Object_list)
                    
                rd.save_results(base_folder_in + str(database_number) + "/" + "data_"+ 
                                str(PPE_p.beta_indx_list[beta_i]) + "_"+ str(PPE_p.alpha_indx_list[alpha_i])+"_EVO", All_Object_list)
    
