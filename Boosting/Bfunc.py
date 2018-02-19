# -*- coding: utf-8 -*-
"""
Created on Tue Mar 24 01:31:58 2015

@author: montoya
"""

import numpy as np
import matplotlib.pyplot as plt
import copy as copy
import time

def set_Classifier (self, classifier):
    # This functions sets the classifier.
    # This classifier must have all its personal parameters set and the methods:
    # classifier.fit(Xtrain, Ytrain, W)
    # classifier.soft_out(X)
    self.base_learner = classifier
    self.learners = [];             # List of the base learners
    
    self.outMode = classifier.outMode
    self.set_Train(classifier.Xtrain, classifier.Ytrain)
    self.set_Test(classifier.Xtest, classifier.Ytest)
    self.set_Val(classifier.Xval, classifier.Yval)

def set_T (self, T):
    self.T = T   # BEST T of the booster
    self.Tmax = T  # Maximum T
    
def get_O (self, X, tEnd = -1):
    # Gets the output of the system for the first tEnd learners

    if (tEnd < -0.5):
        tEnd = self.T 
#        print tEnd
    
    Nsamples, Ndim = X.shape;
    output = np.zeros((Nsamples,1))
    
    for t in range (tEnd):
        output += self.learners[t].get_O(X) * self.alphas[t];

#        print output
#        print self.alphas[t]
    return output

###########################################################################################################################
#########################################  Obtain resulting network ##########################################################
###########################################################################################################################

def get_SLFN (self,tEnd = -1):
    if (tEnd < -0.5):
        tEnd = self.T 
        
    total_SLFN = copy.deepcopy(self.learners[0]);
    total_Wh = self.learners[0].Wh
    total_bh = self.learners[0].bh
    total_Wo = self.learners[0].Wo
    total_bo = self.learners[0].bo
    total_nh = self.learners[0].nh
    
    for t in range (1,tEnd):
        total_Wh = np.concatenate((total_Wh,self.learners[t].Wh), axis = 1)
    
    
#        print output
#        print self.alphas[t]
    return output


import interface
class Boosting_param:
    
    # Class with the important parametes of the NN for results and conclussions
    def __init__(self, myBust):
                     
        #  MSE: Mean Square Error
        #  Cross-Entropy: NLL of probability (output)
        self.base_learner = interface.CSLFN_param(myBust.learners[-1])
        self.T = myBust.Tmax
        self.alg = myBust.alg
        
def output_stuff(self, dir_folder, params = ["File"]):
    # This function stores the parameters of the NN into a file.
    # The name of the file is the value of the parameters given in params[]
    # separated by _

    Param_obj = Boosting_param(self)
    Exec_obj = interface.Execution_param(self)
    
    # Store results of the experiments !! 
    import pickle_lib as pkl
    
    # Create name
    name = str('');
    for param in params:
        name += str(param)+"_"
        
    pkl.store_pickle(dir_folder + name,[Exec_obj, Param_obj],1)
    
    # This function is in charge of outputing all the parameters of the network
    # for a single mingle set of parameters in the training set.


def evaluate_stop(self, e, error_tr, error_val):
    
    if (self.alphas[e] < 0.01):
        return 1
    
    return 0
    

###########################################################################################################################
######################################### sklearn INTEFACE !!  ##########################################################
###########################################################################################################################
def instant_score(self,O,Y):
    predicted = np.sign(O)
    N_samples,nO = Y.shape
    score = 0.0;

    for i in range (N_samples):
#            print predicted[i], Y[i]
        if (Y[i] == predicted[i]):
            score += 1;
    
    score /= N_samples;
    return score;
    
def predict_proba(self,X, tEnd = -1):
    O = self.get_O(X, tEnd)    # Get the output of the net
#        print O
    return O
        
def predict(self,X,tEnd = -1):
    O = self.predict_proba(X, tEnd)
    predicted = np.sign(O)
    return predicted
            
def score(self,X,Y,tEnd = -1):
    
    predicted = self.predict(X, tEnd)
    N_samples,nO = Y.shape
    score = 0.0;

    for i in range (N_samples):
#            print predicted[i], Y[i]
        if (Y[i] == predicted[i]):
            score += 1;
    
    score /= N_samples;
    return score;
    
    