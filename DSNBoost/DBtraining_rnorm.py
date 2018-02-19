# -*- coding: utf-8 -*-
"""
Created on Tue Mar 24 01:31:58 2015

@author: montoya
"""

import numpy as np
import copy as copy

def fi(self, l,O):
    # This function transforms the propagated O when WholeO
#    O = O/np.sum(self.gammas[:l])
    O = O/np.max(np.abs(O))
    
#    O = np.zeros(O.shape)
    
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
    
    self.gammas = np.zeros((nL,1))    
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
            print 'BDSN Layer: '+ str(l) + "/"+str(nL)
    
        layer = copy.deepcopy(self.base_layer)  # Copy base layer
        
        Xlayer = Xtrain         # Total training input of the layer 
        XlayerVal = Xval        # Total validation imput of the layer
        
        if (l != 0):            # Concatenate previous outputs !! 
            if (self.Inyection == "PrevZo"):
                Xlayer = np.concatenate((Xtrain, Zo), axis = 1)  # Build training data
                XlayerVal = np.concatenate((Xval, ZoVal), axis = 1)  # Build validation data
    
            elif (self.Inyection == "WholeO"):
                Xlayer = np.concatenate((Xtrain,self.fi(l,O)), axis = 1)  # Build training data
                XlayerVal = np.concatenate((Xval,self.fi(l,Oval)), axis = 1)  # Build validation data
    
#        print Xlayer.shape
        layer.set_Train(Xlayer, Ytrain)         # Set the training data
        layer.set_Val(XlayerVal, Yval)         # Set the training data
        
        layer.init_Weights()                    # Init weights randomly
        
        # Make the initialiization of the propagation closer to 1  
#        if (l != 0):
#            # Wh = ni x nh 
##            layer.Wh[-1,:] = np.random.uniform(0,1,(1,layer.nH))    
#            layer.Wh[-1,:] = np.ones((1,layer.nH)) 

#        print layer.Wh.shape
#        print W
        layer.set_D(W);   # Set samples distribution
        layer.train_once();                     # SOLO UN PUTO ENTRENAMIENTO
        
        # Set Previous output for the next layer
        Zo = layer.get_Zo(Xlayer)
        ZoVal = layer.get_Zo(XlayerVal)
        
        #############################
        """ Zo Normalization """
        #############################
        if (layer.fo.__name__ == "linear"):
        # Normalize output so that Zo out is between -1 and 1
            max_Zo = np.max(np.abs(Zo))
            
            if (self.visual[0] == 1):
                print "Normalized Zo " + str(max_Zo)
            
            layer.Wo = layer.Wo/max_Zo
            layer.bo = layer.bo/max_Zo
            
            Zo = Zo / max_Zo
            ZoVal = ZoVal / max_Zo
        
        #################################################################
        ######### OBTAIN INYECTION (agregation with realAdaBoost)  ######
        #################################################################

        Ol = layer.fo(Zo)               # Output of the layer
        OlVal = layer.fo(ZoVal)
        
        if (self.Agregation == "GentleBoost"):
             self.gammas[l] = 1;    # For the GentleBoost
        
        elif (self.Agregation == "RealAdaBoost"):
            r = np.sum(W * Ytrain * Ol);
            if (np.abs(r) > 1):         # Check that r is valid
                print "Puto R mayor que 1"
                exit(-1)

            self.gammas[l] = np.log((1+r)/(1-r))/2;
            print "Gamma " + str(self.gammas[l]) +  " R = "+ str(r)
            
        if (l == 0):
            O = r*Ol        # update strong classifier
            Oval = r*OlVal 
        else:
            O = r*Ol + (1-r)*O/np.abs(np.max(O));       # update strong classifier
            Oval = r*OlVal + (1-r)*Oval/np.abs(np.max(Oval))

###     Dont normalize output every time :(
#        max_O = np.max(np.abs(O))
#        O = O/max_O
#        Oval = Oval/max_O
        
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

#             Ol   Onorm
#            print np.sum(W)
            for i in range(W.shape[0]):
                if (W[i] < 0):
                    W[i] = 0
                    print "Ya la estamos liando: Probabilidad Negativa !!"
                    
        if (self.Enphasis == "RealAdaBoost"):
            W = W * np.exp(-Ytrain * Ol * self.gammas[l]);
#            W = np.exp(-Ytrain * Onorm);
            # This is actually the same as the previous but normaling the total output I m affraid.
            # This softens the weigts, maked them more similar.
            
#        print r

        # Add the new learner to the structure
        self.layers.append(layer)
    
####################################################################
####################################################################
####################################################################
    
        """ CALCULATE VISUAL SHIT """
        if (self.visual.store_layers_scores == 1):
            self.scoreTr_layers[l] = layer.instant_score(O, self.Ytrain)
            self.scoreVal_layers[l] = layer.instant_score(Oval, self.Yval)
            print "Score Tr: "+ str(self.scoreTr_layers[l]) + " Score Val" + str(self.scoreVal_layers[l]) 
            
        if (self.visual.store_layers_soft_error == 1):   
            self.errorTr_layers[l] = layer.soft_error(O/np.abs(np.max(O)), self.Ytrain)
            self.errorVal_layers[l] = layer.soft_error(Oval/np.abs(np.max(Oval)), self.Yval)

####################################################################
####################################################################
####################################################################
        
    ################# STOP CHECKING #################
        
#        stop = self.check_stop_L(l, n_gammas = 2)
        stop = 0    # Never stop !!
        
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

def check_stop_L(self, l,n_gammas = 5):
    # Checks if it is time to stop stacking networks
    stop = 0;
    
    if (self.Enphasis == "RealAdaBoost"):  # If gammas is small enough
        if (self.gammas[l] < 0.01):
            stop = 1
            
    if (self.Enphasis == "NeoBoosting"):
        # If the sum of all gammas is 10 times bigger than the last 10 gammas
    
        method = 0

        if (method == 0):
            if (l >= 2 * n_gammas):    ### SETS THE MINIMUM AMOUNT OF LAYERS !!

                ini = np.max((0,(l+1)-n_gammas))
                ini2 = np.max((0,(l+1)-2*n_gammas))
                
#                print ini, ini2
                last_n_gammas = np.average(self.gammas[ini:l+1])
                prev_last_n_gammas = np.average(self.gammas[ini2:ini])
            
#                print self.gammas[ini:l+1].shape, self.gammas[ini2:ini].shape
    
                C = 0.005
#                print "Current n, Prev N"
#                print str(np.concatenate((self.gammas[ini:l+1],self.gammas[ini2:ini]), axis = 1))
                Returns = np.abs((last_n_gammas - prev_last_n_gammas)/last_n_gammas)
                Returns = Returns 
                
                print Returns
                if (Returns < C):
                    stop = 1
                
        if (method == 1):
            n_gammas = 5
            Comp = 2
            
            all_gammas = np.sum(self.gammas[:l+1])
            ini = np.max((0,l-n_gammas))
            last_10_gammas = np.sum(self.gammas[ini:l+1])
            
    #        print all_gammas, last_10_gammas
    
            
            print "All: " + str(all_gammas) + ", Ten last: " + str(Comp * last_10_gammas)
            if (all_gammas > Comp * last_10_gammas ):
                stop = 1
            
    return stop;
    
def get_O (self, X, lEnd = -1):
    # Gets the output of the system for the first tEnd learners
    if (lEnd < -0.5):
        lEnd = self.nL
#        print tEnd
    
    Nsamples, Ndim = X.shape;

    Zo = []
    O = np.zeros((Nsamples,1))
    for l in range (lEnd):
        if (l == 0):
            Zo = self.layers[l].get_Zo(X);
        else:
            if (self.Inyection == "PrevZo"):
                Xlayer = np.concatenate((X, Zo), axis = 1) 
                
            elif (self.Inyection == "WholeO"):
                Xlayer = np.concatenate((X, self.fi(l,O)), axis = 1) 
                
#            print Xlayer.shape, l
#            print self.layers[l+1].Wh.shape
#            print self
            Zo = self.layers[l].get_Zo(Xlayer);
        
        r = (np.exp(2*self.gammas[l]) -1)/(1 + np.exp(2*self.gammas[l]))
        if (l == 0):
            O = r*self.layers[l].fo(Zo)
        else:
            O = r*self.layers[l].fo(Zo) + (1-r)*O/np.abs(np.max(O));

#        print output
#        print self.alphas[t]
    return O


def set_L (self, L):
    self.nL = L   # BEST T of the booster
    self.Lmax = L  # Maximum T


import interface
class DSNBoost_param:
    
    # Class with the important parametes of the NN for results and conclussions
    def __init__(self, myDSNBust):
                     
        #  MSE: Mean Square Error
        #  Cross-Entropy: NLL of probability (output)
        self.base_learner = interface.CSLFN_param(myDSNBust.layers[-1])
        self.Lmax = myDSNBust.Lmax
        self.nL = myDSNBust.nL
        self.alpha = myDSNBust.alpha
        self.beta = myDSNBust.beta
        
        self.Inyection = myDSNBust.Inyection
        self.Agregation = myDSNBust.Agregation
        self.Enphasis = myDSNBust.Enphasis
        
        # Intermediate results info.
        self.gammas = myDSNBust.gammas
        
        if (myDSNBust.visual.store_layers_scores == 1):
            self.scoreTr_layers = myDSNBust.scoreTr_layers
            self.scoreVal_layers = myDSNBust.scoreVal_layers
            
        if (myDSNBust.visual.store_layers_soft_error == 1):   
            self.errorTr_layers = myDSNBust.errorTr_layers
            self.errorVal_layers = myDSNBust.errorVal_layers
            
def output_stuff(self, dir_folder, params = ["File"]):
    # This function stores the parameters of the NN into a file.
    # The name of the file is the value of the parameters given in params[]
    # separated by _

    Param_obj = DSNBoost_param(self)  # This is the last DSNBoost trained
    Exec_obj = interface.Execution_param(self)
    
    # Store results of the experiments !! 
    import pickle_lib as pkl
    
    # Create name
    name = str('');
    for param in params:
        name += str(param)+"_"
        
    pkl.store_pickle(dir_folder + name,[Exec_obj, Param_obj],1)
    


    