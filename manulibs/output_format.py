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


def generate_table_results (aves, stds, params, param_list):
    # It generates a 2 D table with the results of it.
    # Aves and std are 1 Dimensional. 
    # Params is 2 Dimensional and it contains the parameters associated to the ave and std.
    # For example [0.4, 0.8]
    # Param_list contains all possible parameters types.

    # For every Sample in "params" we find its position in "param_list" and put its value in
    # the right place !!


    N_samples = aves.flatten().length
    
    # Obtain the size of the matrix
    list_size = len(param_list)
    list_dim = []
    for i in range (list_size):
        list_dim.append(param_list[i].flatten().length)
        
    Results_Matrix = np.ones((list_dim))  # Create the results matrix
    Results_Matrix = Results_Matrix * -1  # If there are unfilled values then they are -1
    
    for i in range(N_samples): # For every samples !!
    
        ind1 = np.where(param_list[0] == params[0])
        ind2 = np.where(param_list[1] == params[1])
        
        if (aves[i] > Results_Matrix[ind1,ind2] ):  # If the BEST result so far.
            Results_Matrix[ind1,ind2] = aves[i]
    
    return Results_Matrix
    
    
def print_HTML_table(data, row_length):
    print '<table>'
    counter = 0
    for element in data:
        if counter % row_length == 0:
            print '<tr>'
        print '<td>%s</td>' % element
        counter += 1
        if counter % row_length == 0:
            print '</tr>'
    if counter % row_length != 0:
        for i in range(0, row_length - counter % row_length):
            print '<td>&nbsp;</td>'
        print '</tr>'
    print '</table>'