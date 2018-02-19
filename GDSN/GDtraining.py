import numpy as np
import copy as copy
import SLFN


def train_once (self):
    nL = self.nL
    Xtrain = self.Xtrain
    Ytrain = self.Ytrain
    
    Xval = self.Xval
    Yval = self.Yval
    
    
    Nsamples, Ndim = Xtrain.shape;
    NsamplesVal, Ndim = Xval.shape;
    
    if (self.visual[0] == 1):
        scoreTr = np.zeros((nL,1))
        scoreVal = np.zeros((nL,1))
        errorTr = np.zeros((nL,1))
        errorVal = np.zeros((nL,1))
    
    PrevZo = [];         # Previous outputs of the system for training
    PrevZoVal = [];      # Previous outputs of the system for validation
    
    ############# TRAINING OF EACH SLFN ##############
    """ We propagate only the activation of the output, not the output,
    the next layer will apply "fg" and do whatever he wants with the propagation"""
    
    self.layers = []  # Empty layers !!
    
    for l in range (nL):  # For every layer
#        print 'Layer: '+ str(l) + "/"+str(nL)

        if (l == 0):   
            # If it is the first layer, then it is an SLFN instead of an GSLFN
            # because we dont have anything to propagate.
            # We convert the GSLFN into an SLFN
            
            # Maybe create a contructor copy !!!
            layer = SLFN.CSLFN();
            layer.nH = self.base_layer.nH
            layer.fh = self.base_layer.fh
            layer.dfh = self.base_layer.dfh
            layer.trainingAlg = self.base_layer.trainingAlg
            layer.initDistrib = self.base_layer.initDistrib
            
        else:
            layer = copy.deepcopy(self.base_layer)  # Copy base layer
                    
            ####################################
            ### ADD NOISE TO PROPAGATION #######
            ####################################
                    
#            PrevZo = PrevZo + self.propNoise(PrevZo, ["normal",0, 0.005])
#             TO BE REMOVE
            layer.set_GTrain(PrevZo)   # Set retroalimentation data
            layer.set_GVal(PrevZoVal)  # Set retroalimentation data

        layer.set_Train(Xtrain, Ytrain)         # Set the training data
        layer.set_Val(Xval, Yval)               # Set the validation data
        
        layer.init_Weights()                    # Init weights randomly
        layer.train_once();                     # Train this motherfucker just once !!


        if (l == 0):
            # Set Previous output for the next layer
            PrevZo = layer.get_Zo(Xtrain)
            PrevZoVal = layer.get_Zo(Xval)
            

        else:
            # Set Previous output for the next layer
            PrevZo = layer.get_Zo(Xtrain, PrevZo)
            PrevZoVal = layer.get_Zo(Xval,PrevZoVal)
            
        if (self.visual[0] == 1):
                
            O = layer.fo(PrevZo)
            Oval = layer.fo(PrevZoVal)
        
            scoreTr[l] = layer.instant_score(O, self.Ytrain)
            scoreVal[l] = layer.instant_score(Oval, self.Yval)
    
            errorTr[l] = layer.soft_error(O, self.Ytrain)
            errorVal[l] = layer.soft_error(Oval, self.Yval)
                
#                print self.get_MSE(PrevZo,self.Ytrain)
        # Add the new learner to the structure  
        self.layers.append(layer)
        
    print scoreTr[-1], scoreVal[-1]
    if (self.visual[0] == 1):
        self.manage_results(nL, scoreTr,scoreVal,errorTr,errorVal)
    
import Ginterface
class GDSN_param:
    
    # Class with the important parametes of the NN for results and conclussions
    def __init__(self, myGDSN):
                     
        #  MSE: Mean Square Error
        #  Cross-Entropy: NLL of probability (output)
        self.base_learner = Ginterface.CGSLFN_param(myGDSN.layers[-1])
        self.nL = myGDSN.nL
        self.nP = myGDSN.nP

import interface
def output_stuff(self, dir_folder, params = ["File"]):
    # This function stores the parameters of the NN into a file.
    # The name of the file is the value of the parameters given in params[]
    # separated by _

    Param_obj = GDSN_param(self)
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


