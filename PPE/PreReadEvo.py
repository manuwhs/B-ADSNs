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

""" THIS CODE READS THE RESULTS OF THE CV cluster and also the omniscient
and displays the best CrossValidated Results"""

def PPE_read(PPE_p, main_folder,database_number, repetitions = 20): 
    # Folder where we read the RAW files
    folder = "../"+main_folder +"/"+str(database_number)
    # Folder where we read and store the Preread files
    base_folder_in = "../"+main_folder +"/PreRead/"   

    # We will build a file for every pair [beta, alpha]
    # Every file will contain the "N_realizations" for different values of [Nepoch, Nh]
    """ $$$$$$$$$$$$  Evolution Error  $$$$$$$$$$$$$$$$ """
    
    mu.create_dirs(base_folder_in + str(database_number) + "/")
    
    if (PPE_p.alpha_indx_list[0] == 10):

        All_Object_list = []  # List that will contain of the object lists for a given [beta_alpha]
        All_Exec_list = []   
        alpha_i = 0
        beta_i = 0
        print "WE are delaing with the one an only 10"
        for n_epoch_i in range(len(PPE_p.N_epochs_indx_list)): 
            
            All_Object_list.append([])
            All_Exec_list.append([])
            
            for nH_i in range(len(PPE_p.nH_indx_list)):
                
                Parameters = [PPE_p.nH_indx_list[nH_i],PPE_p.N_epochs_indx_list[n_epoch_i],
                                   PPE_p.fo_list[0],
                                   1,PPE_p.BatchSize_list[0],
                                   PPE_p.Inyection_list[0],PPE_p.Enphasis_list[0],
                                   PPE_p.alpha_indx_list[alpha_i],PPE_p.beta_indx_list[beta_i],
                                   PPE_p.CV_List[0],PPE_p.Nruns_List[0]]
                        
                if (PPE_p.CV_List[0] != 0):  # If we are not loading evolution, but old style final results                    
                    Exec_list = rd.load_Exec_repetitions(folder, [Parameters], repetitions)
#                            print Exec_list
                    All_Exec_list[n_epoch_i].append(Exec_list)
                else:
                    Object_list = rd.load_objects_repetitions(folder, [Parameters], repetitions)
                    All_Object_list[n_epoch_i].append(Object_list)
        
        # We write them all b = 0,1,2,.... copying the original alpha = 10, beta = 0
        for beta_i in range(len(PPE_p.beta_indx_list)):
            if (PPE_p.CV_List[0] != 0):  # If we are not loading evolution, but old style final results                        
                rd.save_results(base_folder_in + str(database_number) + "/" + "data_"+ 
                                str(PPE_p.beta_indx_list[beta_i]) + "_"+ str(PPE_p.alpha_indx_list[alpha_i])+"_EVO", All_Exec_list)
            else:
                rd.save_results(base_folder_in + str(database_number) + "/" + "data_"+ 
                                str(PPE_p.beta_indx_list[beta_i]) + "_"+ str(PPE_p.alpha_indx_list[alpha_i])+"_EVO", All_Object_list)

        return 5
    
    # For evety combination of fucking useless parameters:
    for alpha_i in range(len(PPE_p.alpha_indx_list)):
        for beta_i in range(len(PPE_p.beta_indx_list)):
            
#            # If the alpha == 1, then beta does not mind and it is 10
#            if (PPE_p.alpha_list[PPE_p.alpha_indx_list[alpha_i]] == 1):
#                beta_i = 5
                        
            ## First check if the file with the read data already exists
#            results = pkl.load_pickle(base_folder_in + str(database_number) + "/" + "data_" + 
#                      str(PPE_p.beta_indx_list[beta_i]) + "_"+ str(PPE_p.alpha_indx_list[alpha_i])+"_EVO",1)  # If the result file existis coz it was previously read
            results = []
            read_flag = 1
            
            if ((results != [])&(read_flag == 0)):
                All_Object_list = results
                
            else:
                All_Object_list = []  # List that will contain of the object lists for a given [beta_alpha]
                                      # It will contain All_Object_list[Nepoch][Nh] = N_realizations objects
                All_Exec_list = []    # Se usa solo para los que no tienen evolucion
                # Get the list of parameters and load them into All_Object_list
    
                for n_epoch_i in range(len(PPE_p.N_epochs_indx_list)): 
                    
                    All_Object_list.append([])
                    All_Exec_list.append([])
                    
                    for nH_i in range(len(PPE_p.nH_indx_list)):
                        
                        Parameters = [PPE_p.nH_indx_list[nH_i],PPE_p.N_epochs_indx_list[n_epoch_i],
                                           PPE_p.fo_list[0],
                                           1,PPE_p.BatchSize_list[0],
                                           PPE_p.Inyection_list[0],PPE_p.Enphasis_list[0],
                                           PPE_p.alpha_indx_list[alpha_i],PPE_p.beta_indx_list[beta_i],
                                           PPE_p.CV_List[0],PPE_p.Nruns_List[0]]
                                           
                        if (PPE_p.CV_List[0] != 0):  # If we are not loading evolution, but old style final results                    
                            Exec_list = rd.load_Exec_repetitions(folder, [Parameters], repetitions)
#                            print Exec_list
                            All_Exec_list[n_epoch_i].append(Exec_list)
                        else:
                            Object_list = rd.load_objects_repetitions(folder, [Parameters], repetitions)
                            All_Object_list[n_epoch_i].append(Object_list)
                            
                if (PPE_p.CV_List[0] != 0):  # If we are not loading evolution, but old style final results                        
                    rd.save_results(base_folder_in + str(database_number) + "/" + "data_"+ 
                                    str(PPE_p.beta_indx_list[beta_i]) + "_"+ str(PPE_p.alpha_indx_list[alpha_i])+"_EVO", All_Exec_list)
                else:
                    rd.save_results(base_folder_in + str(database_number) + "/" + "data_"+ 
                                    str(PPE_p.beta_indx_list[beta_i]) + "_"+ str(PPE_p.alpha_indx_list[alpha_i])+"_EVO", All_Object_list)
