# -*- coding: utf-8 -*-
"""
Created on Tue Mar 24 01:31:58 2015

@author: montoya
"""
import numpy as np
from sklearn.lda import LDA
import copy as copy

def LDA_train (self, param):
    H = self.get_Htotal(self.Xtrain,self.GXtrain )   # Get the hidden output matrix
    
    ##############  BOOSTING  ##############
    if (self.D_flag == 1):   # If we have given Weights to the samples
        W_root = np.sqrt(self.D)
        H = H * W_root

    
    lda = LDA(solver = 'lsqr')   # svd , lsqr, eigen
    lda.fit(H,self.Ytrain.ravel())

    self.bo = np.zeros((1,self.nO))  # In the standard ELM, these does not count
    
    proyection = lda.coef_

    self.Wo = proyection.T                   # Write the output weights into the structure

    Hmeans = copy.deepcopy(lda.means_)
    self.Hmeans = Hmeans  ### CREATE IT INTERNAL TO USE SOMEWERE ELSE
    self.priors = copy.deepcopy(lda.priors_)
    threshold = np.dot(proyection, (Hmeans[0,:]+ Hmeans[1,:])/2) - np.log(self.priors[1]/self.priors[0])
#        print Hmeans.shape
#        print proyection.shape
    self.bo = -threshold
    
#    print lda.score(self.H, self.Ytrain.ravel())
#    print self.score(self.Xtrain, self.Ytrain)
#    print "***************"
        
    
def LDAT_train (self, param, D = []):
    # Training using the FineTuning algorithm which is a mixture of ELM and BMBP


    n_epoch = param[0]
    step = param[1]
    partitions = param[2]
    # SET THE OUTPUT LAYER
    self.LDA_train(D = D)
    self.BMBP_train([n_epoch, step, partitions], D = D)
    
    