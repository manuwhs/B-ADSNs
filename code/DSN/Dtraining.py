# -*- coding: utf-8 -*-
"""
Created on Tue Mar 24 01:31:58 2015

@author: montoya
"""

import numpy as np
import copy as copy

def fi(self, O):
    # This function transforms the propagated O when WholeO
#    O = O/np.sum(self.gammas[:l])
    O = O/np.max(np.abs(O))

    return O
#    O = np.zeros(O.shape)
    
def train_once (self):
    # Trains the DSN layer by layer.
    # Each layer is a SLFN object trained itself.

    
    ############# PARAMETERS ##############
    nL = self.nL
    Xtrain = self.Xtrain
    Ytrain = self.Ytrain
    
    Xval = self.Xval
    Yval = self.Yval
    
    Nsamples, Ndim = Xtrain.shape;
    NsamplesVal, Ndim = Xval.shape;
    
    if (self.visual.store_layers_scores == 1):
        self.scoreTr_layers = np.zeros((nL,1))
        self.scoreVal_layers = np.zeros((nL,1))
        
    if (self.visual.store_layers_soft_error == 1):   
        self.errorTr_layers = np.zeros((nL,1))
        self.errorVal_layers = np.zeros((nL,1))
    
    PrevZos = [];         # Previous outputs of the system for training
    PrevZoVals = [];      # Previous outputs of the system for validation
    
    
    """ Imaginary for the stopping cond """
    self.gammas = np.zeros((nL,1))    
    W  = np.ones((Nsamples,1));   # Weight of each sample
    
    ############# TRAINING OF EACH SLFN ##############
    """ We propagate only the activation of the output, not the output,
    the next layer will apply "fg" and do whatever he wants with the propagation"""
    
    self.layers = []  # Empty layers !!

    for l in range (nL):  # For every layer
        # Train weak learner
#        print 'Layer: '+ str(l) + "/"+str(nL)
    
        W = W / np.sum(W);  # Normalize weights
        if (self.visual.verbose == 1):
            print 'DSN Layer: '+ str(l) + "/"+str(nL)
            
        layer = copy.deepcopy(self.base_layer)  # Copy base layer
        
        Xlayer = Xtrain         # Total training input of the layer 
        XlayerVal = Xval        # Total validation imput of the layer
        
        if (l != 0):            # Concatenate previous outputs !! 
#            print PrevZos.shape
            
            Xlayer = np.concatenate((Xtrain,PrevZos), axis = 1)  # Build training data
            XlayerVal = np.concatenate((Xval, PrevZoVals), axis = 1)  # Build validation data
        
        layer.set_Train(Xlayer, Ytrain)         # Set the training data
        layer.set_Val(XlayerVal, Yval)         # Set the training data
        
        layer.init_Weights()                    # Init weights randomly
        layer.train_once();                     # SOLO UN PUTO ENTRENAMIENTO
        
        # Set Previous outputs for the next layer
        if (l == 0): 
            PrevZos = self.fi(layer.get_Zo(Xlayer))
            PrevZoVals = self.fi(layer.get_Zo(XlayerVal))
            
        else:
            """ get_Zo or get_O """
            PrevZos = np.concatenate((PrevZos, self.fi(layer.get_Zo(Xlayer))), axis = 1)
            PrevZoVals = np.concatenate((PrevZoVals, self.fi(layer.get_Zo(XlayerVal))), axis = 1)
            
            if (l >= self.nP):   # IF we have reached the total output inyection number
                PrevZos = np.delete(PrevZos, 0, axis = 1)
                PrevZoVals = np.delete(PrevZoVals, 0, axis = 1)
                
        # Add the new learner to the structure
        self.layers.append(layer)
        
        #### $$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$ ###
        """ Get the alpha for the stop checking """
        O = self.fi(layer.fo(PrevZos[:,-1]))    # Last output
#        print W
        r = np.sum(W.flatten() * Ytrain.flatten() * O.flatten());
        self.gammas[l] = np.log((1+r)/(1-r))/2;
#        print r, self.gammas[l]
        
        #### $$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$ ###
        
        
        Oval = layer.fo(PrevZoVals[:,-1])
        
####################################################################
####################################################################
####################################################################
    
        """ CALCULATE VISUAL SHIT """
        if (self.visual.store_layers_scores == 1):
            self.scoreTr_layers[l] = layer.instant_score(O, self.Ytrain)
            self.scoreVal_layers[l] = layer.instant_score(Oval, self.Yval)
            print "Score Tr: "+ str(self.scoreTr_layers[l]) + " Score Val" + str(self.scoreVal_layers[l]) 
            
        if (self.visual.store_layers_soft_error == 1):   
            self.errorTr_layers[l] = layer.soft_error(O, self.Ytrain)
            self.errorVal_layers[l] = layer.soft_error(Oval, self.Yval)

####################################################################
####################################################################
####################################################################

    ################# STOP CHECKING #################
        
#        stop = self.check_stop_L(l)
        stop = 0
        if (stop == 1): #
            self.nL = l +1
            print "Stoped Earlier at L: " + str(self.nL )
            self.gammas = self.gammas[:self.nL]
            
            if (self.visual.store_layers_scores == 1):
                self.scoreTr_layers = self.scoreTr_layers[:self.nL]
                self.scoreVal_layers = self.scoreVal_layers[:self.nL]
                
            if (self.visual.store_layers_soft_error == 1):   
                self.errorTr_layers = self.errorTr_layers[:self.nL]
                self.errorVal_layers = self.errorVal_layers[:self.nL]
                
            break
    
    if (self.visual.plot_results_layers == 1):
        self.manage_results(self.nL, self.scoreTr_layers,self.scoreVal_layers, self.errorTr_layers, self.errorVal_layers)



def check_stop_L(self, l):
    # Checks if it is time to stop stacking networks
    stop = 0;
    
        # If the sum of all gammas is 10 times bigger than the last 10 gammas
    
    gammas_real = copy.deepcopy(self.gammas)
    
    if (l >= 5):  # JEJEJEJEJEJE
        n_gammas = 2
    
        ini = np.max((0,(l+1)-n_gammas))
        ini2 = np.max((0,(l+1)-2*n_gammas))
        
#                print ini, ini2
        last_n_gammas = np.average(gammas_real[ini:l+1])
        prev_last_n_gammas = np.average(gammas_real[ini2:ini])
    
#                print self.gammas[ini:l+1].shape, self.gammas[ini2:ini].shape

        C = 0.005
#                print "Current n, Prev N"
#                print str(np.concatenate((self.gammas[ini:l+1],self.gammas[ini2:ini]), axis = 1))
        Returns = np.abs((last_n_gammas - prev_last_n_gammas)/last_n_gammas)
        Returns = Returns * n_gammas
        
        print Returns
        if (Returns < C):
            stop = 1
                
            
    return stop;



import interface
class DSN_param:
    
    # Class with the important parametes of the NN for results and conclussions
    def __init__(self, myDSN):
                     
        #  MSE: Mean Square Error
        #  Cross-Entropy: NLL of probability (output)
        self.base_learner = interface.CSLFN_param(myDSN.layers[-1])
        self.nL = myDSN.nL
        self.nP = myDSN.nP
        
        # Intermediate results info.
        # Intermediate results info.
        self.gammas = myDSN.gammas
        
        if (myDSN.visual.store_layers_scores == 1):
            self.scoreTr_layers = myDSN.scoreTr_layers
            self.scoreVal_layers = myDSN.scoreVal_layers
            
        if (myDSN.visual.store_layers_soft_error == 1):   
            self.errorTr_layers = myDSN.errorTr_layers
            self.errorVal_layers = myDSN.errorVal_layers
        
def output_stuff(self, dir_folder, params = ["File"]):
    # This function stores the parameters of the NN into a file.
    # The name of the file is the value of the parameters given in params[]
    # separated by _

    Param_obj = DSN_param(self)
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
