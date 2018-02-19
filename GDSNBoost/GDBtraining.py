# -*- coding: utf-8 -*-
"""
Created on Tue Mar 24 01:31:58 2015

@author: montoya
"""

import numpy as np
import copy as copy
import SLFN

def fi(self, l,O):
    # This function transforms the propagated O when WholeO
#    O = O/np.sum(self.gammas[:l+1])
    O = O/np.max(np.abs(O))
    
    return O

def fi2(self, l,O):
#    O = np.zeros(O.shape)
    O = O/np.max(np.abs(O))
    return O
    
def train_once (self):
    # Trains the DSN layer by layer.
    # Each layer is a SLFN object trained itself.

    ############# PARAMETERS ##############
    nL = self.Lmax
    Xtrain = self.Xtrain
    Ytrain = self.Ytrain
    
    Xval = self.Xval
    Yval = self.Yval
    
    Nsamples, Ndim = Xtrain.shape;
    NsamplesVal, Ndim = Xval.shape;
    
    self.gammas = np.ones((nL,1))    
    W  = np.ones((Nsamples,1));   # Weight of each sample
    
    O = np.zeros((Nsamples,1));  # Output of the whole system for every sample
    Oval = np.zeros((NsamplesVal,1));  # Output for validation (faster than calling .score() all the time)
    

    if (self.visual.store_layers_scores == 1):
        self.scoreTr_layers = np.zeros((nL,1))
        self.scoreVal_layers = np.zeros((nL,1))
        
    if (self.visual.store_layers_soft_error == 1):   
        self.errorTr_layers = np.zeros((nL,1))
        self.errorVal_layers = np.zeros((nL,1))
    
    Zo = [];         # Previous outputs of the system for training
    ZoVal = [];      # Previous outputs of the system for validation
    
    ############# TRAINING OF EACH SLFN ##############
    """ We propagate only the activation of the output, not the output,
    the next layer will apply "fg" and do whatever he wants with the propagation"""
    
    self.layers = []  # Empty layers !!

    for l in range (nL):  # For every layer
        W = W / np.sum(W);  # Normalize weights
        
        if (self.visual.verbose == 1):
            print 'BGDSN Layer: '+ str(l) + "/"+str(nL)
    
        layer = copy.deepcopy(self.base_layer)  # Copy base layer
        
        if (l == 0):   
            # If it is the first layer, then it is an SLFN instead of an GSLFN
            # because we dont have anything to propagate.
            # We convert the GSLFN into an SLFN
            
            layer = SLFN.CSLFN();
            layer.nH = self.base_layer.nH
            layer.fh = self.base_layer.fh
            layer.dfh = self.base_layer.dfh
            layer.trainingAlg = self.base_layer.trainingAlg
            layer.initDistrib = self.base_layer.initDistrib
        else:
            layer = copy.deepcopy(self.base_layer)  # Copy base layer
            
            if (self.Inyection == "PrevZo"):
                layer.set_GTrain(Zo)   # Set retroalimentation data
                layer.set_GVal(ZoVal)  # Set retroalimentation data
#                print "Zo", Zo
                
            elif (self.Inyection == "WholeO"):
                layer.set_GTrain(self.fi(l,O))   # Set retroalimentation data
                layer.set_GVal(self.fi(l,Oval))  # Set retroalimentation data

#        print Xlayer.shape
        layer.set_Train(Xtrain, Ytrain)         # Set the training data
        layer.set_Val(Xval, Yval)         # Set the training data
        
#        print "JODER"
#        print l
        layer.init_Weights()                    # Init weights randomly
#        print layer.Wh.shape
#        print W
        layer.set_D(W);   # Set samples distribution
        layer.train_once();                     # SOLO UN PUTO ENTRENAMIENTO
        
        if (l >0):
            print "Weights of output"
            print layer.Wo[layer.nH-2:,:]   ## Print the weight of the G neuron
        # Set Previous output for the next layer
        if (l == 0):
            Zo = layer.get_Zo(Xtrain)
            ZoVal = layer.get_Zo(Xval)
        else:
            # Set Previous output for the next layer
            # The layer.fg is applied here because we have to use
            # the "fg" from the current layer !!!!!!!
        
            if (self.Inyection == "PrevZo"):
                Zo = layer.get_Zo(Xtrain, Zo)
                ZoVal = layer.get_Zo(Xval,ZoVal)

#                print "Zo", Zo
            elif (self.Inyection == "WholeO"):
                Zo = layer.get_Zo(Xtrain, self.fi(l,O))
                ZoVal = layer.get_Zo(Xval,self.fi(l,Oval))
        
        #############################
        """ Zo Normalization """
        #############################
        # Normalize output so that Zo out is between -1 and 1
        if (layer.fo.__name__ == "linear"):
            max_Zo = np.max(np.abs(Zo))
            
#            if (self.visual[0] == 1):
#                print "Normalized Zo " + str(max_Zo)
            
            layer.Wo = layer.Wo/max_Zo
            layer.bo = layer.bo/max_Zo
            
            Zo = Zo / max_Zo
            ZoVal = ZoVal / max_Zo
        
        #################################################################
        ######### OBTAIN INYECTION (agregation with realAdaBoost)  ######
        #################################################################

        Ol = layer.fo(Zo)               # Output of the layer
        OlVal = layer.fo(ZoVal)
            
        # Obtain the dehenphasis alpha of the learner given by RealAdaboost
        
        if (self.Agregation == "GentleBoost"):
             self.gammas[l] = 1;    # For the GentleBoost
             
        elif (self.Agregation == "RealAdaBoost"):
            r = np.sum(W * Ytrain * Ol);
            if (np.abs(r) > 1):         # Check that r is valid
                print "Puto R mayor que 1"
                exit(-1)

            self.gammas[l] = np.log((1+r)/(1-r))/2;
            if (self.visual.verbose == 1):
                print "Gamma " + str(self.gammas[l]) +  " R = "+ str(r)
            
        O += Ol * self.gammas[l];       # update strong classifier
        Oval += OlVal * self.gammas[l]
        
        ########### RECALCULATE NEW ENPHASIS #################3
        # Maybe we want to calculate the enfasis according to the whole system and the alpha according to the last layer....
        if (l == 0):
            Onorm = Ol / np.max(np.abs(Ol))
        else:
            Onorm = O / np.max(np.abs(O))
            
        if (self.Enphasis == "NeoBoosting"):
            # Either use the whole Ol or the O total normalized !! Onorm
            Anibals_constant = 4                                       ### Anibal say it is 4 but it does not work !!
            W = self.alpha + (1 - self.alpha) * \
            (self.beta * np.power(Onorm - self.Ytrain,2)/Anibals_constant + \
            (1 - self.beta) * (1 - np.power(Onorm,2)))

#            print np.sum(W)
            for i in range(W.shape[0]):
                if (W[i] < 0):
                    W[i] = 0
                    print "Ya la estamos liando: Probabilidad Negativa !!"
                    
        if (self.Enphasis == "RealAdaBoost"):
            W = W * np.exp(-Ytrain * Ol * self.gammas[l]);
#            W = W * np.exp(-Ytrain * O * self.gammas[l]);
#        print r
#        print self.gammas[l]
        if (self.Enphasis == "ManuBoost"):
            Wrab = ( W * np.exp( (self.beta * (np.power(Ol - self.Ytrain,2)) -
                            (1 - self.beta )* (np.power(Ol,2))) * self.gammas[l]))
            Wrab = Wrab / np.sum(Wrab)
#            print Wrab.size
            W = (float(self.alpha)/Wrab.size)*np.ones(Wrab.shape) + (1 - self.alpha )
        # Add the new learner to the structure
        self.layers.append(layer)
        
    ################# STOP CHECKING #################
    
#        if (self.gammas[l] < 0.01):
#            self.nL = l +1
#            if (self.visual[0] == 1):
#                scoreTr = scoreTr[:self.nL]
#                scoreVal = scoreVal[:self.nL]
#        
#                errorTr = errorTr[:self.nL]
#                errorVal = errorVal[:self.nL]
#            
#            break
        
        """ CALCULATE VISUAL SHIT """
        if (self.visual.store_layers_scores == 1):
            self.scoreTr_layers[l] = layer.instant_score(O, self.Ytrain)
            self.scoreVal_layers[l] = layer.instant_score(Oval, self.Yval)
            if (self.visual.verbose == 1):
                print "Score Tr: "+ str(self.scoreTr_layers[l]) + " Score Val" + str(self.scoreVal_layers[l]) 
            
        if (self.visual.store_layers_soft_error == 1):   
            self.errorTr_layers[l] = layer.soft_error(O/np.abs(np.max(O)), self.Ytrain)
            self.errorVal_layers[l] = layer.soft_error(Oval/np.abs(np.max(Oval)), self.Yval)

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

def get_O (self, X, lEnd = -1):
    # Gets the output of the system for the first lEnd layers
    """ It only propagates the Activation of outputs """
    
    if (lEnd < - 0.5):
        lEnd = self.nL 
#        print lEnd
    
    Nsamples, Ndim = X.shape;
    O = np.zeros((Nsamples,1))
    
    for l in range (lEnd):
        if (l == 0):
            Zo = self.layers[l].get_Zo(X);
        else:
            if (self.Inyection == "PrevZo"):
                Zo = self.layers[l].get_Zo(X,Zo);
            elif (self.Inyection == "WholeO"):
                Zo = self.layers[l].get_Zo(X,self.fi(l,O));
    # We apply the last output right
        O += self.layers[l].fo(Zo) * self.gammas[l]
    
#        print output
    return O
        

def set_L (self, L):
    self.nL = L   # BEST T of the booster
    self.Lmax = L  # Maximum T
    


import interface
import Ginterface
class GDSNBoost_param:
    
    # Class with the important parametes of the NN for results and conclussions
    def __init__(self, myGDSNBust):
                     
        #  MSE: Mean Square Error
        #  Cross-Entropy: NLL of probability (output)
        self.base_learner = Ginterface.CGSLFN_param(myGDSNBust.layers[-1])
        self.Lmax = myGDSNBust.Lmax
        self.nL = myGDSNBust.nL
        self.alpha = myGDSNBust.alpha
        self.beta = myGDSNBust.beta
        
        self.Inyection = myGDSNBust.Inyection
        self.Agregation = myGDSNBust.Agregation
        self.Enphasis = myGDSNBust.Enphasis
        
        self.gammas = myGDSNBust.gammas  # alphas os the adaboost
        
def output_stuff(self, dir_folder, params = ["File"]):
    # This function stores the parameters of the NN into a file.
    # The name of the file is the value of the parameters given in params[]
    # separated by _

    Param_obj = GDSNBoost_param(self)
    Exec_obj = interface.Execution_param(self)
    
    # Store results of the experiments !! 
    import pickle_lib as pkl
    
    # Create name
    name = str('');
    for param in params:
        name += str(param)+"_"
        
    pkl.store_pickle(dir_folder + name,[Exec_obj, Param_obj],1)
    
    
    