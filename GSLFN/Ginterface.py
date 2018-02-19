# -*- coding: utf-8 -*-
"""
Created on Tue Mar 24 01:31:58 2015

@author: montoya
"""

import numpy as np



###########################################################################################################################
######################################### sklearn INTEFACE !!  ##########################################################
###########################################################################################################################

def fit (self, Xtrain, XGtrain, Ytrain):
    self.set_Train (Xtrain, Ytrain)
    self.set_GTrain (XGtrain)
    self.train()

def predict_proba(self,X,Xg):
    O = self.get_O(X,Xg)    # Get the output of the net
    return O
    
def soft_out(self,X,Xg):
    O = self.get_O(X,Xg)    # Get the output of the net
    return O
    
def predict(self,X,Xg):
    O = self.predict_proba(X,Xg)

    if (self.nO == 1):      # Standard binary classification


        if (self.fo.__name__ == "sigm"):  # sigmoid goes from 0 to 1
            predicted = np.zeros((len(O)))
            for i in range (len(O)):
                if (O[i] > 0.5):
                    predicted[i] = 1
        else:  # tanh and linear goes from -1 to 1
            predicted = np.sign(O)
            
    else:                   # Multiclass classification
        predicted = np.argmax(O, axis = 1 ) # Obtain for every sample, the class with the highest probability
#        print predicted
    return predicted;
    
def score(self,X,Xg,Y):
    
    predicted = self.predict(X,Xg)
    N_samples,nO = Y.shape
    score = 0.0;
#        print N_samples
#        print X.shape
#        
    if (self.nO == 1):
        for i in range (N_samples):
#                print predicted[i], Y[i]
            if (Y[i] == predicted[i]):
                score += 1;
    
    score /= N_samples;
    return score;

class CGSLFN_param:
    
    # Class with the important parametes of the NN for results and conclussions
    def __init__(self, NN):
                     
        #  MSE: Mean Square Error
        #  Cross-Entropy: NLL of probability (output)
        self.nH = NN.nH                      # Number of hidden neurons 
                                             # Activation functions of hidden and output neurons
        self.errFunc = NN.errFunc            # Error Function (some algorihms impose this function)
  
        self.nG = NN.nG
        
        self.fh_name = NN.fh.func_name
        self.fo_name = NN.fo.func_name
        self.fg_name = NN.fg.func_name
        
        self.trainingAlg = NN.trainingAlg           # Training Algorithm
        self.initDistrib = NN.initDistrib           # Distribution for the initialization
        self.regularization = NN.regularization     # Regularization
        self.stopCrit = NN.stopCrit                 #
        
import interface
def output_stuff(self, dir_folder, params = ["File"]):
    # This function stores the parameters of the NN into a file.
    # The name of the file is the value of the parameters given in params[]
    # separated by _
    Param_obj = CGSLFN_param(self)
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

    



