# -*- coding: utf-8 -*-
"""
Created on Tue Mar 24 01:31:58 2015

@author: montoya
"""

import numpy as np


class CGMDH:
    
    def __init__(self, Niter = 5, M = 10):
        # Niter: Number of Iterations of expansion
        # M: Number of expanded Zij we keep for the next expansion
        # If M is an integer, all expansing will have the same M
        # If it is a list of size Niter, then thats the M for every iteration
        
        self.Niter =  Niter                       # Number of hidden neurons 
        self.M = M 

    ####################################
    """ WEIGHTS INITILIZATION """
    ###################################
    def fit (self,Xtrain, Ytrain):
        
        Nsamples, Ndim = Xtrain.shape
        
        # If we are given a number instead of a list
        if (type(self.M) is int ):
            self.M = np.ones(self.Niter) * self.M  # Number of selected Zij
        self.M = self.M.astype(int)
        
        Niter = self.Niter      # Number of iterations 
        
        self.Selected_ij = []        # Niter list with the index i and j of the slected Xi, Xj
        self.W_ij = []               # Niter list with the weight vector of the selected Zij
        
        Z = Xtrain              # Initialize the Z 
        
        Nsamples, Ndim = Xtrain.shape
#        print M
        
        for e in range (Niter):     ### FOR EVERY ITERATION LAYER ###
            
            # Obtain the number of dimensions of input and output space
            if (e == 0):
                Ndim_in = Ndim
                print Ndim_in
            else:
                Ndim_in = self.M[e -1]
            
            Ndim_out = self.M[e]
            
            # Initialize structure
            self.Selected_ij.append([])   # Append empty list for the M[e] selected i,j
            self.W_ij.append([])           # Append empty list for the M[e] calculated Wij
            
            # Obtain the optimal Zij for every (m over 2)  i,j combination storing
            #      - Emse
            #      - ij
            #      - wij
            
            
            Emse = np.zeros (((Ndim_in*(Ndim_in-1))/2,1))
#            print Emse.shape
            
            W_ij_iter = []          # Matrix [Niter][M] with the extensions of the Zs
            Selected_ij_iter = []   # Matrix [Niter][2] with the source variables Xi, Xj
            
            indx = 0
            
            ## Obtain the MSE of every expansion 
            for j in range (1 , Ndim_in):
                for i in range (0, j):
                    Z_spand = self.get_expansion (Z[:,i], Z[:,j])
                    Selected_ij_iter.append([i,j])
                    
                    Wijaux = self.get_Wij(Z_spand, Ytrain)
                    W_ij_iter.append(Wijaux);
                    
                    Emse[indx] = self.get_MSE (Z_spand, Wijaux, Ytrain)
                    
                    indx += 1
            # Order the expansions by MSE 
            Emse_ordered, Emse_order = self.sort_and_get_order(Emse)
#            print Emse_ordered
            
            ###### Select a subset of them #####
            
            ### BEST individual subset ###
            
            Znext_indx = []
            for i in range (Ndim_out):
                Znext_indx.append(Emse_order[i])
                
#            print Znext
#            print Emse_ordered[0]
#            print Emse[Znext[0]]
            ##### STORE DATA STRUCTURES OF THE SPACE #######
            """ The BEST Zij is the first one !!!!"""
            
            for z in Znext_indx:
                self.Selected_ij[e].append(Selected_ij_iter[z])
                self.W_ij[e].append(W_ij_iter[z])
            
            self.best_Zfinal = 0
            ##### BUILD NEXT INPUT SPACE ######
            ## We only need to build it if there is going to be another iteration
            Z = self.get_Znext(self.W_ij[e],  self.Selected_ij[e], Z)
            
#            print Z.shape
###########################################################################################################################
######################################### sklearn INTEFACE !!  ##########################################################
###########################################################################################################################

    def get_expansion(self, Xi, Xj ):
        # Given 2 input unidimensional variables, this function outputs the polinomial 
        # extension of order 2 of them for all their samples. (column vectors)
        
        Nsamples = len(Xi)
        Xi = Xi.reshape((Nsamples,1))
        Xj = Xj.reshape((Nsamples,1))
        
        X_spand = np.concatenate((np.ones((Xi.shape)), Xi*Xj,
                                  Xi, Xj,
                                  np.power(Xi,2), np.power(Xj,2)), axis = 1)
#        print X_spand.shape
        
        return X_spand
    
    def get_Wij(self, X, T):  
        # Obtain Least MSE  W vector solution of the input X
       
        Xinv = np.linalg.pinv(X)         # Get the inverse of the matrix 
#        print Xinv.shape
#        print T.shape
        Wij = np.dot(Xinv,T)  # Get output weights
        
        return Wij
    
    def get_MSE(self, X, W, T):  
        # Obtain the Least MSE solution of given expansion
        
        Nsamples, Ndim = X.shape
        Y = np.dot(X,W)  # Get output weights
#        print X.shape
#        print W.shape
        
        error = np.sum(np.power(Y-T,2)) / Nsamples
        return error
        
    def sort_and_get_order (self,x):
        # Sorts x in increasing order and also returns the ordered index
    
        order = range(len(x.flatten()))
        x_ordered, order = zip(*sorted(zip(x, order)))
        return np.array(x_ordered), np.array(order)
        
    def get_Znext (self,Wij_list, Selected_ij_list, Zprev):
        # You give him the previous input space and the XiXj combinations and it calculates
        # the new input space using the already calculated Wij weigt vectors
    
        M = len(Wij_list)
        Nsamples, Ndim = Zprev.shape
        
        Znext = np.zeros((Nsamples,M))
        for m in range(M):  # For every i,j combination of Zprev
#            print Selected_ij_list[m]
            
            Zijexpand = self.get_expansion(Zprev[:,Selected_ij_list[m][0]],  Zprev[:,Selected_ij_list[m][1]] )
#            print Zijexpand.shape, Wij_list[m].shape
            
            Zij = np.dot(Zijexpand, Wij_list[m])
            
            Znext[:,m] = Zij.flatten()
            
        return Znext
        
    def predict (self, X):
        # Once trained, this function outputs the result for each training sample in X
        Z = X
        Nsamples, Ndim = X.shape
        for e in range (self.Niter):   # For every iteration 
            ##### BUILD NEXT INPUT SPACE ######
            Z = self.get_Znext(self.W_ij[e],  self.Selected_ij[e], Z)
#            print Z[:,self.best_Zfinal]
            
        y = Z[:,self.best_Zfinal]    # The Best Zij of the final step is the first one

        return y.reshape(Nsamples,1)
        
    def score (self, X, Y):
        Nsamples, Ndim = X.shape
        
        O = self.predict(X)
        score_MSE = np.sum(np.power(Y-O,2)) /Nsamples

        return score_MSE



###########################################################################################################################
######################################### visual SHIT !!  ##########################################################
###########################################################################################################################
 
    def set_visual (self, param = [0]):
        # Function sets the visualization parameters for obtaining the evolution
        # of intermediate results.
        self.visual = param;
    