# -*- coding: utf-8 -*-
"""
Created on Tue Mar 24 01:31:58 2015

@author: montoya
"""

""" SAME AS THE SLFN """

from sklearn.cross_validation import StratifiedKFold  # For crossvalidation
import numpy as np
import matplotlib.pyplot as plt

def ELM_train (self, param, D = []):
    type_bias = param[0]
    
    G = self.get_G(self.Xtrain)     # Get the hidden output matrix
    
    ##############  BOOSTING  ##############
    if (D != []):   # If we have given Weights to the samples
        W_root = np.sqrt(D)
        G = G * W_root
    
    if (type_bias == "no_bias"):
        self.bo = np.zeros((1,self.nO))  # In the standard ELM, these does not count
        
        Ginv = np.linalg.pinv(G)         # Get the inverse of the matrix 
        beta = np.dot(Ginv,self.Ytrain)  # Get output weights
        self.Wo = beta                   # Write the output weights into the structure
    
    if (type_bias == "bias"):
        
        Nsa, Nh = G.shape
        G = np.concatenate((G, np.ones((Nsa,1))),axis = 1)   # Add a bias value to every sample
        Ginv = np.linalg.pinv(G)                # Get the inverse of the matrix 
        beta = np.dot(Ginv,self.Ytrain)         # Get output weights and bias
        
        self.Wo = beta[:Nh:]                 # Obtain the output weights
        self.bo = beta[Nh,:]                 # Obtain the output bias

#        print self.bo
        

def ELM_validate (self, nC, param, n_iter = 10, D = []):
    # nH is the the list of nH values to validate
    # Validating values shoudld have been given.
    nParam = len(nC)
    scoreTr = np.zeros((nParam,1))
    scoreVal = np.zeros((nParam,1))
    
    for i in range (nParam):       # For every possible value of nH
        print "Using "+ str(nC[i]) + "/"+ str(nC[-1]) + " Centers"
        for j  in range (n_iter):  # Average over a number of tries
            self.set_nC(nC[i])   # Set new number of neurons
            self.init_Centers () # Reintilize weights
            self.ELM_train(param, D);    # Train the net
            scoreTr[i] += self.score(self.Xtrain, self.Ytrain)
            scoreVal[i] += self.score(self.Xval, self.Yval)
        scoreTr[i] /= n_iter
        scoreVal[i] /= n_iter
    
    best_indx = np.argmax(scoreVal, axis = 0 )
    best_nC = nC[best_indx]
    
    plt.figure()
    plt.plot(nC,scoreTr, lw=3)
    plt.plot(nC,scoreVal, lw=3)
    plt.title('Accuracy ELM')
    plt.xlabel('N hidden neurons')
    plt.ylabel('Accuracy')
    plt.legend(['Train','Test'])
    plt.grid()
    plt.show()
    
    return (best_nC, scoreTr, scoreVal)
        
        