# -*- coding: utf-8 -*-
"""
Created on Tue Mar 24 01:31:58 2015

@author: montoya
"""

def check_stop (gammas, n_gammas = 5):
    
    ng = gammas.size
    
    n_gammas = n_gammas  # Window size of gammas
    
    C = 0.005
    
    for l in range (ng):
        if (l >= 2 *n_gammas ):    ### SETS THE MINIMUM AMOUNT OF LAYERS !!
            ini = np.max((0,(l+1)-n_gammas))
            ini2 = np.max((0,(l+1)-2*n_gammas))
            
    #                print ini, ini2
            last_n_gammas = np.average(gammas[ini:l+1])
            prev_last_n_gammas = np.average(gammas[ini2:ini])
        
    #                print self.gammas[ini:l+1].shape, self.gammas[ini2:ini].shape
    
    #                print "Current n, Prev N"
    #                print str(np.concatenate((self.gammas[ini:l+1],self.gammas[ini2:ini]), axis = 1))
            Returns = np.abs((last_n_gammas - prev_last_n_gammas)/last_n_gammas)
            Returns = Returns # Maybe multiplied by  n_gammas 
            
            if (Returns < C):   # Stability check
                return l+1
                
                
            if (last_n_gammas < C*10): # Low value check
                return l+1
                
    return ng  # If it does not stop before the max
    
import os
import numpy as np
import import_folders
import_folders.imp_folders(os.path.abspath(''))
import manutils as mu
import results_reader as rd
import matplotlib.pyplot as plt
import pickle_lib as pkl

plt.close("all")
""" THIS CODE READS THE INTERMEDIATE RESULTS OF THE LAYERS AND PLOTS THEM"""
""" FOR NOW ONLY ONE RUN PER FILE WITH NO CV !!!!! """ 


#################  PARAMETROS DEL BARRIDO ##################
## Here we introduce the init and end values of the cluster
## These are constant
fo_list = [0]
Learning_Rate_list = [0]
BatchSize_list = [3]
Inyection_list =  [0]
Enphasis_list = [1]
CV_List = [0]     # 1 is for training with all dataset
Nruns_List = [0]

### These are the ones we actually search
nH_list = [2,4,6,8,10,12,14,16,18,20,22,24,26,28,30,32,34,36,38,40,42,44,46,48]
nH_indx_list = [0,2,4,6,8,10,12,14,16,18,20,22,24,26,28,30,32,34,36,38,40,42,44,46]

N_epochs_list = [50,100,200]
N_epochs_indx_list = [0,1,2]

alpha_list = [0,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1]
alpha_indx_list = [0,1,2,3,4,5,6,7,8,9,10]

beta_list = [1]
beta_indx_list = [10]

#################  GENERATE PARAMTERS SETS ##################

database_number = 2
beta_i = 0
alpha_i = 0


# Folder where we read the RAW files
folder = "/NeoEvo/ResultsNeoDSN"+str(database_number)
# Folder where we read and store the Preread files
base_folder_in = "../NeoEvo/PreRead/"   
# Folder where we store the graph
base_folder_out = "../NeoEvo/Gold/"   

results = pkl.load_pickle(base_folder_in + str(database_number) + "/" + "data_" + "_"+ 
                            str(beta_indx_list[beta_i]) + "_"+ str(alpha_indx_list[alpha_i])+"_EVO",1)  # If the result file existis coz it was previously read

if (results == []):
    print "FILE NOT PREREAD"
    exit(-1)

All_Object_list = results

Nepoch = len(All_Object_list[0])
N_neurons = len(All_Object_list[0][0])

nL_max = 200  # Maximum number of layers
# Obtain the score of training and validation for every layer and realization
All_scoreTr_layers = []
All_scoreVal_layers = []
for i in range(N_neurons):
    scoreTr_layers, scoreVal_layers = rd.get_scores_layers (All_Object_list[i])
    All_scoreTr_layers.append(scoreTr_layers)
    All_scoreVal_layers.append(scoreVal_layers)
    
# Get the average value and std for all the layers of the validation score
    
All_aves_rea_val = []
All_stds_rea_val = []

for i in range(N_neurons):
    matrix = mu.convert_to_matrix(All_scoreVal_layers[i],nL_max)
    aves_rea_val, stds_rea_val = mu.get_ave_std_unfilled_matrix(matrix)

    All_aves_rea_val.append(aves_rea_val)
    All_stds_rea_val.append(stds_rea_val)
    
# Get the average value and std for all the layers of the training score
All_aves_rea_tr = []
All_stds_rea_tr = []

for i in range(N_neurons):
    matrix = mu.convert_to_matrix(All_scoreTr_layers[i],nL_max)
    aves_rea_tr, stds_rea_tr = mu.get_ave_std_unfilled_matrix(matrix)

    All_aves_rea_tr.append(aves_rea_tr)
    All_stds_rea_tr.append(stds_rea_tr)
    

# Get the gammas values and their average and std
All_gammas = []
N_realizations = []

All_aves_gammas = []
All_stds_gam = []

for i in range(N_neurons):
    
    gammas = rd.get_gammas (All_Object_list[i])    
    All_gammas.append(gammas)
    N_realizations.append(len(gammas))
    
    matrix = mu.convert_to_matrix(gammas)
    aves_gam, stds_gam = mu.get_ave_std_unfilled_matrix(matrix)

    All_aves_gammas.append(aves_gam)
    All_stds_gam.append(stds_gam)
    


""" 1rd GRAPH """ 
# Plot the Average Training and Validation score !!

nLs_list= range (nL_max)
# Plot the average and shit
mu.plot_all_tr_val_nL(nLs_list, All_aves_rea_val, All_aves_rea_tr)

nLs_list= range (nL_max)
plt.savefig(base_folder_out + str(database_number) +"/" + "Ave_Accu(nL,nH)_"
            + str(N_epochs_list[Nepoch_i])+".png")
plt.close("all")

""" 2rd GRAPH """ 
# Plot the gammas evolution

""" 3rd GRAPH """ 
# Plot the Validation accuracy as function of gamma

""" 4th GRAPH """

# Write the resulting Average and shit of using n_gammas = 5
# for all number of neurons


ave_val = np.ones((N_neurons,1))
std_val = np.ones((N_neurons,1))

ave_NLs = np.ones((N_neurons,1))
std_Nls = np.ones((N_neurons,1))

for nh_i in range(N_neurons):

    # Obtain nLs
    nLs = np.ones(N_realizations[nh_i])
    accuracies = np.ones((N_realizations[nh_i],1))
    
    for j in range(N_realizations[nh_i]):  # For every realization
        nLs[j] = check_stop(All_gammas[nh_i][j],5 )
        
    # Get the NLs statistics
    ave_NLs[nh_i] = np.mean(nLs)
    std_Nls[nh_i] = np.std(nLs)
    
    for j in range (N_realizations[nh_i]):  # For every realization
        pene = All_scoreVal_layers[nh_i][j]
        accuracies[j] = pene[nLs[j]-1]
        
    ave_val[nh_i] = np.mean(accuracies)
    std_val[nh_i] = np.std(accuracies)
    
mu.plot_results_ngamma_accu(nH_list,ave_val,std_val)

plt.savefig(base_folder_out + str(database_number) +"/" + "Ave_Accu(nH) ng=5_"
            + str(N_epochs_list[Nepoch_i])+".png")
            
plt.close("all")

mu.plot_results_ngamma_nL(nH_list, ave_NLs, std_Nls )
plt.savefig(base_folder_out + str(database_number) +"/" + "Ave_nLs(nH) ng=5_"
            + str(N_epochs_list[Nepoch_i])+".png")
plt.close("all")

""" 2rd GRAPH """ 
# Plot the Average Training and Validation score !!

mu.plot_all_tr_val_nL_surface(nH_list, nLs_list, All_aves_rea_val, All_aves_rea_tr)
plt.savefig(base_folder_out + str(database_number) +"/" + "3D_plot_Ave_Accu(nH,nL)"
            + str(N_epochs_list[Nepoch_i])+".png")
            
plt.close("all")

