# -*- coding: utf-8 -*-
"""
Created on Tue Mar 24 01:31:58 2015

@author: montoya
"""

import numpy as np
import matplotlib.pyplot as plt

###########################################################################################################################
######################################### sklearn INTEFACE !!  ##########################################################
###########################################################################################################################

def fit (self, Xtrain, Ytrain):
    self.set_Train (Xtrain, Ytrain)
    self.train()

def predict_proba(self,X):
    O = self.get_O(X)    # Get the output of the net
    return O
    
def soft_out(self,X):
    O = self.get_O(X)    # Get the output of the net
    return O
    
def predict(self,X):
    O = self.predict_proba(X)
    if (self.nO == 1):      # Standard binary classification
    
        if (self.outMode == -1):  # tanh goes from -1 to 1
            predicted = np.sign(O)
            
        if (self.outMode == 0):  # sigmoid goes from 0 to 1
            predicted = np.zeros((np.alen(O),1)) 
            for i in range (np.alen(O)):
                if (O[i] > 0.5):
                    predicted[i] = 1

    else:                   # Multiclass classification
        predicted = np.argmax(O, axis = 1 ) # Obtain for every sample, the class with the highest probability
#        print predicted
    return predicted;
    
def score(self,X,Y):
    
    predicted = self.predict(X)
    N_samples,nO = Y.shape
    score = 0.0;
#        print N_samples
#        print X.shape
#        
    if (self.nO == 1):
        for i in range (N_samples):
#            print predicted[i], Y[i]
            if (Y[i] == predicted[i]):
                score += 1;
    else: # Multiclass case
        for i in range (N_samples):
#            print predicted[i], Y[i]
            if (Y[i,predicted[i]] == 1):
                score += 1;
                
    score /= N_samples;
    return score;

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
        
def soft_error (self, O, T):
    
    error = 0;
    if (self.errFunc == "MSE"):
        error = self.get_MSE (O, T)
            
    elif (self.errFunc == "EXP"):
        error = np.average(np.exp(-T * O))
            
    elif (self.errFunc == "CE"):
        O = O + 1/10000000   # So that division by O is not 0
        error = np.average( T * np.log(O) + (1-T) * np.log(1-O))
            
    return error
    
def get_MSE (self, O, T):
    error = np.power((O-T),2)
    error = np.average(error)
    return error
        
def manage_results (self,n_epoch, scoreTr,scoreVal,errorTr,errorVal):
        plt.figure()
        plt.plot(range(n_epoch),scoreTr, lw=3)
        plt.plot(range(n_epoch),scoreVal, lw=3)
        plt.title('Accuracy BP. step =')
        plt.xlabel('Epoch')
        plt.ylabel('Accuracy')
        plt.legend(['Train','Test'])
        plt.grid()
        plt.show()
    
        plt.figure()
        plt.plot(range(n_epoch),errorTr, lw=3)
        plt.plot(range(n_epoch),errorVal, lw=3)
        plt.title('Error BP. step =')
        plt.xlabel('Epoch')
        plt.ylabel('Accuracy')
        plt.legend(['Train','Test'])
        plt.grid()
        plt.show()

        
import time 
class CSLFN_param:
    
    # Class with the important parametes of the NN for results and conclussions
    def __init__(self, NN):
                     
        #  MSE: Mean Square Error
        #  Cross-Entropy: NLL of probability (output)
        self.nH = NN.nH                      # Number of hidden neurons 
                                             # Activation functions of hidden and output neurons
        self.fh_name = NN.fh.func_name
        self.fo_name = NN.fo.func_name
        
        self.errFunc = NN.errFunc            # Error Function (some algorihms impose this function)
  
        self.trainingAlg = NN.trainingAlg           # Training Algorithm
        self.initDistrib = NN.initDistrib           # Distribution for the initialization
        self.regularization = NN.regularization     # Regularization
        self.stopCrit = NN.stopCrit                 #
        

class Execution_param:
    def __init__(self, Exec):
        self.Nruns = Exec.Nruns           # Number of times the network is reinitialized
        self.CV = Exec.CV                 # K of the Kfold crossvalidation
        
        self.TrError = Exec.TrError           # Training Error of the Nruns
        self.ValError = Exec.ValError         # Validation Error of the Nruns
        self.TstError = Exec.TstError         # Validation Error of the Nruns    
        
        self.RandomSeed = Exec.RandomSeed   # Random seed of every CV so that we can redo it again
    
        self.Date = time.strftime("%c")
        
def output_stuff(self, dir_folder, params = ["File"]):
    # This function stores the parameters of the NN into a file.
    # The name of the file is the value of the parameters given in params[]
    # separated by _
    Param_obj = CSLFN_param(self)
    Exec_obj = Execution_param(self)
    
    # Store results of the experiments !! 
    import pickle_lib as pkl
    
    # Create name
    name = str('');
    for param in params:
        name += str(param)+"_"
        
    pkl.store_pickle(dir_folder + name,[Exec_obj, Param_obj],1)
    
    # This function is in charge of outputing all the parameters of the network
    # for a single mingle set of parameters in the training set.

    






