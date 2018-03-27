# -*- coding: utf-8 -*-
"""
Created on Tue Mar 24 01:31:58 2015

@author: montoya
"""
from sklearn.cross_validation import StratifiedKFold  # For crossvalidation
import numpy as np
import matplotlib.pyplot as plt

import os.path

import pickle_lib as pkl


########################################################################
################### Reading Repetition functions ######################
#######################################################################
def fuse_rep (file_rep):
    # It fusions all the repetitions of a task in an array !!!

    Obj = file_rep[0][1]   # We get any of the objects (rhe firts one mismamente)
    Num_rep = len(file_rep)
#    print "Numer rep " + str(Num_rep)
    Tr_list = []
    Val_list = []
    Tst_list = []
    
    for i in range(Num_rep):  # We extract all the Tr Val and Tst Values
        Tr_list.append(file_rep[i][0].TrError)
        Val_list.append(file_rep[i][0].ValError)
        Tst_list.append(file_rep[i][0].TstError)
    
#    print "Pene " + str(Tr_list)
    Exec = file_rep[0][0]    # Provisional equalling
    Exec.TrError = np.array(Tr_list)
    Exec.ValError = np.array(Val_list)
    Exec.TstError = np.array(Tst_list)
#    print Exec.ValError
    return Exec, Obj
    
def load_results_repetitions(folder_name, params_list = [], repetitions = 1):
    
    # Loads the results that have been donde in various repetitions in the cluster
    # params is a bidimensional list where each row [i,:] is a list
    # of all the parameters that were crossvalidated !!

    Object_list = []    # List with the paramters of the simulation
    Exec_list = []      # Object with the results of the simulation
    
    for paramS in params_list:
        
        file_name = str('')
        
        if type(paramS) is list:
            for param in paramS:
                file_name += str(param)+"_"
        
        else:    # If there was only one parameter validaded
            file_name += str(paramS)+"_"
            
        
        path_obj = "./"+ folder_name+"/" + file_name
        
        ### NOW WE INTRODUCE THE REPETITION PART !!!!
        
        obj_rep = []
        for i in range (repetitions):
            path_obj_rep = path_obj + str(i) +"_" 
            obj = pkl.load_pickle(path_obj_rep ,1)
            
            if(obj == []): # The object does not exist
                print path_obj + " does not exist"
            else:
                obj_rep.append(obj)
        
        if (obj_rep != []): # IF there was any file of the repetition
            Exec, Object = fuse_rep(obj_rep);
                
            Exec_list.append(Exec)
            Object_list.append(Object)   # The 0 is because pickle only saves lists 
                                        # we saved the object as a list.
    return Exec_list, Object_list



def load_objects_repetitions(folder_name, params_list = [], repetitions = 1):
    
    # Loads the objects (for intermediate results) of the param list.
    # The difference is that we do get all the objects of one param_list
    # instead of only getting one of them.
    # It is meant to be for just one object configuration

    Object_list = []    # List with the paramters of the simulation

    for paramS in params_list:
        
        file_name = str('')
        
        if type(paramS) is list:
            for param in paramS:
                file_name += str(param)+"_"
        
        else:    # If there was only one parameter validaded
            file_name += str(paramS)+"_"
            
        
        path_obj = "./"+ folder_name+"/" + file_name
        
        ### NOW WE INTRODUCE THE REPETITION PART !!!!
        
        Object_list = []
        for i in range (repetitions):
            path_obj_rep = path_obj + str(i) +"_" 
            obj = pkl.load_pickle(path_obj_rep ,1)
            
            if(obj == []): # The object does not exist
                print path_obj + " does not exist"
            else:
                Object_list.append(obj[1])
                                        # we saved the object as a list.
    return Object_list
    
def load_Exec_repetitions(folder_name, params_list = [], repetitions = 1):
    
    # Loads the results that have been donde in various repetitions in the cluster
    # params is a bidimensional list where each row [i,:] is a list
    # of all the parameters that were crossvalidated !!

    Object_list = []    # List with the paramters of the simulation
    Exec_list = []      # Object with the results of the simulation
    
    for paramS in params_list:
        
        file_name = str('')
        
        if type(paramS) is list:
            for param in paramS:
                file_name += str(param)+"_"
        
        else:    # If there was only one parameter validaded
            file_name += str(paramS)+"_"
            
        
        path_obj = "./"+ folder_name+"/" + file_name
        
        ### NOW WE INTRODUCE THE REPETITION PART !!!!
        
        obj_rep = []
        for i in range (repetitions):
            path_obj_rep = path_obj + str(i) +"_" 
            obj = pkl.load_pickle(path_obj_rep ,1)
            
            if(obj == []): # The object does not exist
                print path_obj + " does not exist"
            else:
                obj_rep.append(obj)
        
        if (obj_rep != []): # IF there was any file of the repetition
            Exec, Object = fuse_rep(obj_rep);
                
            Exec_list.append(Exec)
            Object_list.append(Object)   # The 0 is because pickle only saves lists 
                                        # we saved the object as a list.
    return Exec_list



########################################################################
################### Get Stuff from Execution list ######################
#######################################################################
    
#def get_execution_results (listas):
#    
#    N_listas = len(listas)
#    N_realizations = listas[0].TrError.size
#    
#    tr = np.zeros((N_realizations,N_listas))
#    val = np.zeros((N_realizations,N_listas))
#    tst = np.zeros((N_realizations,N_listas))
#    # First position is for the training, second validation and 3rd testing
#    for i in range(N_listas):
#        tr[:,i] = listas[i].TrError.flatten()
#        val[:,i] = listas[i].ValError.flatten()
#        tst[:,i] = listas[i].TstError.flatten()
#    
#    return tr, val, tst
    

def get_execution_results (listas):
    
    N_listas = len(listas)
    tr = []
    val = []
    tst = []
    # First position is for the training, second validation and 3rd testing
    for i in range(N_listas):
        tr.append(listas[i].TrError.flatten())
        val.append(listas[i].ValError.flatten())
        tst.append(listas[i].TstError.flatten())
    
    return tr, val, tst


def get_ave_and_std (listas):
    
    N_listas = len(listas)
    aves = np.zeros((N_listas,2))
    stds = np.zeros((N_listas,2))
    
    # First position is for the training, second validation and 3rd testing
    for i in range(N_listas):
        aves[i][0] = np.average(listas[i].TrError)
        aves[i][1] = np.average(listas[i].ValError)
#        aves[i][2] = np.average(listas[i].TstError)
    
        stds[i][0] = np.std(listas[i].TrError)
        stds[i][1] = np.std(listas[i].ValError)
#        stds[i][2] = np.std(listas[i].TstError)
    
    return aves, stds
    
    
########################################################################
################### Get Stuff from Objects list ######################
#######################################################################

def get_gammas (Objects):
    
    N_objects = len(Objects)
    
    gammas = []

    # First position is for the training, second validation and 3rd testing
    for i in range(N_objects):
        gammas.append(Objects[i].gammas)
    
    return gammas

def get_nLs (Objects):
    
    N_objects = len(Objects)
    
    nL = np.zeros((N_objects,1))

    # First position is for the training, second validation and 3rd testing
    for i in range(N_objects):
        nL[i] = Objects[i].nL
    
    return nL

def get_scores_layers (Objects):
    N_objects = len(Objects)
    
    scoreTr_layers = []
    scoreVal_layers = []
    # First position is for the training, second validation and 3rd testing
    for i in range(N_objects):
        scoreTr_layers.append(Objects[i].scoreTr_layers)
        scoreVal_layers.append(Objects[i].scoreVal_layers)
    return scoreTr_layers, scoreVal_layers
    


########################################################################
################### Save Results ######################
#######################################################################

def save_results(filename, li, partitions = 1):
    # Just saves the results as a pickel.
    pkl.store_pickle(filename, li, partitions = 1)
    print "Object: "+filename+ " saved."



########################################################################
################### Load without repetition (obsolete) ######################
#######################################################################

def load_results(folder_name, params_list = []):
    # params is a bidimensional list where each row [i,:] is a list
    # of all the parameters that were crossvalidated !!

    Object_list = []    # List with the paramters of the simulation
    Exec_list = []      # Object with the results of the simulation
    
    for paramS in params_list:
        file_name = str('')
        
        if type(paramS) is list:
            
            for param in paramS:
                file_name += str(param)+"_"
        
        else:    # If there was only one parameter validaded
            file_name += str(paramS)+"_"
            
        
        path_obj = "./"+ folder_name+"/" + file_name
        obj = pkl.load_pickle(path_obj ,1)
        if(obj == []): # The object does not exist
            print path_obj + " does not exist"
        else:
            Exec_list.append(obj[0])
            Object_list.append(obj[1])   # The 0 is because pickle only saves lists 
                                    # we saved the object as a list.
    return Exec_list, Object_list
