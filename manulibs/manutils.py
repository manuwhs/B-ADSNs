

import matplotlib.pyplot as plt
import numpy as np
import results_reader as rd
import os 

def check_stop (gammas, n_gammas = 5):
    
    gammas = np.array(gammas)
    ng = gammas.size
    
    n_gammas = n_gammas  # Window size of gammas
    
    C = 0.00001
    
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
            Returns = np.abs((last_n_gammas - prev_last_n_gammas))
            Returns = Returns # Maybe multiplied by  n_gammas 
            
#            return l-1
#            if (Returns < C):   # Stability check
#                return l+1
            
#            if (l == n_gammas):   # Stability check
#                print "FRR"
#                return l+1     
                
#            if (last_n_gammas < C*10): # Low value check
#                return l+1
#            if (l == 100):
#                return l+1
    return ng  # If it does not stop before the max
    

def sort_and_get_order (x):
    # Sorts x in increasing order and also returns the ordered index
    order = range(len(x.flatten()))
    x_ordered, order = zip(*sorted(zip(x, order)))
    return np.array(x_ordered), np.array(order)

def get_indx(i,l):
    n = i / l
#    if (i % l == 0):
#        n = n -1
    
    return n

def get_all_indx (i,Nb,Na,Ne,Nh):
    n_b = get_indx(i,Nh*Ne*Na)
    i = i - n_b *Nh*Ne*Na
    
    n_a = get_indx(i ,Nh*Ne)
    i = i - n_a *Nh*Ne
    
    n_e = get_indx(i,Nh)
    i = i - n_e *Nh
    
    n_h = i

    return n_b, n_a, n_e, n_h

def get_all_indx_gen (i,list_indx):
#    list_indx = [Nb,Na,Ni,Nr,Ne,Nh]
    
    output_list = []
    
    Num_dim = len(list_indx)
    
    total_size = 1
    for dim in range(1,Num_dim):
        total_size = total_size * list_indx[dim]
        
    for dim in range(1,Num_dim):
        position = get_indx(i, total_size)
        output_list.append(position)
        
        i = i - position * total_size
        total_size = total_size/list_indx[dim]
        
    output_list.append(i)

    return output_list
    
""" A partir de un array simple X = [1 2 3 4 5 6 7 8 9...]
Al hacer reshape en M[n1,n2,n3], se van colocando desde al final al principio.
M[0,0,:] contendra el [1 2 3... hasta n3]
M[0,1,:] contendra el [6 7 8... hasta n3]
"""

def create_dirs (folder):
    if not os.path.exists(folder):
        os.makedirs(folder)
        
def convert_to_matrix (lista, max_size = -1):
    # Converts a list of lists with different lengths into a matrix 
    # filling with -1s the empty spaces 

    Nlist = len(lista)
    
    listas_lengths = []
    
    if (max_size == -1):
        for i in range (Nlist):
            listas_lengths.append(lista[i].size)
        
        lmax = np.max(listas_lengths)
    else:
        lmax = max_size 
        
    matrix = -1 * np.ones((Nlist,lmax))
    
    for i in range (Nlist):
        if (lista[i].size > lmax):
            matrix[i,:lista[i].size] = lista[i][:lmax].flatten()
        else:
            matrix[i,:lista[i].size] = lista[i].flatten()
    
    return matrix



def get_ave_std_unfilled_matrix(matrix):
    # From the matrix of realizations it gets the average and std for each 
    # number of layers.

    Nruns, lmax = matrix.shape
    aves = np.zeros((lmax,1)).flatten()
    stds = np.zeros((lmax,1)).flatten()
    
    realizations = 0
    
    for i in range (lmax): # For every number of layers
        
        indexes = np.where(matrix[:,i] > 0)[0]

        if (len(indexes) == 0): # If there were no data
#            # Equal to the previous
            aves[i] = aves[i-1]
            stds[i] = stds[i -1]
            
        else:
            
            realizations = matrix[indexes,i]
            aves[i] = np.mean(realizations)
            stds[i] = np.std(realizations)
        
    return aves, stds

def get_final_results_Evo(Objects, ngammas = 5):
    # From all the realizations for a set of paramters.
    # This function applies a stop condition and outputs
    # the ave_tr, std_tr, ave_tst, std_tst

    gammas = rd.get_gammas (Objects) 
    
    N_realizations = len(gammas)

#    print N_realizations
    
    scoreTr_layers, scoreVal_layers = rd.get_scores_layers (Objects)

    # Obtain nLs
    nLs = np.ones(N_realizations)
    
    for j in range(N_realizations):  # For every realization
        nLs[j] = check_stop(gammas[j],ngammas)
        
    # Get the NLs statistics
    acc_val = np.ones((N_realizations,1))
    acc_tr = np.ones((N_realizations,1))
    
    for j in range (N_realizations):  # For every realization
        acc_val[j] = scoreVal_layers[j][nLs[j]-1]
        acc_tr[j] = scoreTr_layers[j][nLs[j]-1]
        
    ave_val = np.mean(acc_val)
    std_val = np.std(acc_val)
    
    ave_tr = np.mean(acc_tr)
    std_tr = np.std(acc_tr)

    return ave_tr,std_tr,ave_val,std_val


def get_final_results_CV(Objects):
    # From all the realizations for a set of paramters.
    # This function applies a stop condition and outputs
    # the ave_tr, std_tr, ave_tst, std_tst

#    print N_realizations
    
#    aves, stds = rd.get_ave_and_std (Objects)
#    print Objects
    ave_val = np.average(Objects[0].ValError)
    std_val = np.std(Objects[0].ValError)
    
    ave_tr = np.average(Objects[0].TrError)
    std_tr = np.std(Objects[0].TrError)

    return ave_tr,std_tr,ave_val,std_val