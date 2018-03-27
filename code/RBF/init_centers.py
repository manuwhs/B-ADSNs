# -*- coding: utf-8 -*-
"""
Created on Tue Mar 24 01:31:58 2015

@author: montoya
"""

import numpy as np
from math_func import * # This way we import the functions directly

def random_samples_centers (self, param = [0]):
    X = self.Xtrain
    nC = self.nC
    
    # Initializes centers using random samples
    rnd_idx = np.random.permutation(X.shape[0])[:nC]
    self.centers = [X[i,:] for i in rnd_idx]



from sklearn.cluster import KMeans
def K_means (self, param = [0]):
    X = self.Xtrain
    nC = self.nC
    
    max_iter = param[0]     # Number of iterations for convergence
    n_init = param[1]       # Number of times of reinitialization 
    split_flag = param[2]
    
    if (split_flag == "split"):  # Uses half of the centroids for one class
                           # and the other half for the others.
        print "SPLIT"
        
        # SPLIT Training Data
        for c in range(self.nO): 
            X0 = X([self.Ytrain[c] == 1])
            
    myKmeans = KMeans(n_clusters = nC, 
                      n_init = n_init, 
                      max_iter = max_iter, 
                      tol=0.0001, 
                      precompute_distances='auto', 
                      verbose=0, random_state=None, 
                      copy_x=True, 
                      n_jobs=1)
                      
    myKmeans.fit(X)             # Set the initialization
    self.centers = myKmeans.cluster_centers_




