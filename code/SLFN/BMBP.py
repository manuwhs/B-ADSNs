# -*- coding: utf-8 -*-
"""
Created on Tue Mar 24 01:31:58 2015

@author: montoya
"""
import numpy as np

def BMBP_train (self, param):
    # Training using the BatchMode Backpropagation Algorithm 
    # W is used to give weights to the samples

    n_epoch = param[0]      # Number of maximum epochs
    step = param[1]         # Learning Step Size
    
    PLOT_DATA = self.visual[0]
    SHOW_EPOCHS = self.visual[0]
    
    scoreTr = np.zeros((n_epoch,1))
    scoreVal = np.zeros((n_epoch,1))
    
    errorTr = np.zeros((n_epoch,1))
    errorVal = np.zeros((n_epoch,1))
    
#    print self.D
    
    nTrSa, Ndim = self.Xtrain.shape
    
    if (len(param) >= 3): # If the number of samples is not specified, we do BMBP
        n_MBSa =   param[2]    # Number of samples of the minibatch
    else: 
        n_MBSa = nTrSa      # It not specified, then we do full mode backprop
    
    
    # We divide the learning rate by the number of samples
#    step = float(step)/float(n_MBSa);


    n_partitions = int(nTrSa / n_MBSa )
    rest = nTrSa - n_partitions * n_MBSa
    
    if (rest > 0):          # If it cannot be divided exactly
        n_partitions += 1   # We add the last smaller partition
            
#    print n_partitions, n_partitions * n_MBSa, nTrSa
#    print rest
    
    for i in range (n_epoch): # For every epoch
        
#        print "#######################################"
        if (SHOW_EPOCHS == 1):
            print "Epoch: "+str(i)+"/"+str(n_epoch)
        
        order = np.random.permutation(nTrSa) # Randomize the order of samples

        if (len(param) >= 4): # IF LINEAR DECAY OF THE LEARNING RAYE
            step = param[1]*(n_epoch - i)/n_epoch
#            print step
        
        if (self.trainingAlg.trAlg == "ELMT"):
            self.ELM_train("bias")
        
#        sleping = 1
        for p in range (n_partitions):  # For every partition !!!
            
            end = np.min(((p+1)*n_MBSa,nTrSa))
            
            X = self.Xtrain[order[p*n_MBSa:end]]
            T = self.Ytrain[order[p*n_MBSa:end]]
            
            nSa, ndim = X.shape
            nSa, nO = T.shape
            
            # Obtain activations and outputs needed
            Zh = np.dot(X,self.Wh) + np.tile (self.bh, (nSa,1))
            H = self.fh(Zh)
            Zo = np.dot (H,self.Wo) + np.tile (self.bo, (nSa,1))
            O = self.fo(Zo)
            
            # Obtain derivative of error depending on the cost function
            dError_dOut = (O - T)

            # Obtain the sigma_k for ever output neuron and sigma_j for hidden neurons

            Sk = dError_dOut * self.dfo(Zo)
            Sk = Sk.T

            ##############  BOOSTING  ##############

            if (self.D_flag == 1):   # If we have given Weights to the samples
                Sk = Sk * self.D[order[p*n_MBSa:end]].T  * nTrSa # * n_MBSa

            Sj = np.dot(self.Wo, Sk) * self.dfh (Zh).T
            Sj = Sj.T

                
            # Obtain derivate of the error with respect to the weights
            dErrorWo = np.dot(H.T,Sk.T)  # Nh * No
            dErrorWh = np.dot(X.T,Sj)  # Ni * Nh
            
            number_samples = 1    # end - p*n_MBSa
            
#            print number_samples
            
            """ If number of samples is too big, the values of the net diverge
            and errors appear """
            
            # Modify the weights and bias
            if (self.trainingAlg.trAlg != "ELMT"):
                # When using ELM, the H dies
                self.Wo -=  dErrorWo * step / number_samples
                self.bo -=  np.sum(Sk) * step/ number_samples

                
            self.Wh -=  dErrorWh * step/ number_samples
            self.bh -=  np.sum(Sj) * step/ number_samples
            
            
    if (PLOT_DATA == 1):
        self.manage_results(n_epoch, scoreTr,scoreVal,errorTr,errorVal)