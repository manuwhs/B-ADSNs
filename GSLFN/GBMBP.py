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
    
    n_partitions = int(nTrSa / n_MBSa )
    rest = nTrSa - n_partitions * n_MBSa
    
    if (rest > 0):          # If it cannot be divided exactly
        n_partitions += 1   # We add the last smaller partition
    # We divide the learning rate by the number of samples
#    step = float(step)/float(n_MBSa);
    
    for i in range (n_epoch): # For every epoch
        
        if (SHOW_EPOCHS == 1):
            print "Epoch: "+str(i)+"/"+str(n_epoch)
        
        order = np.random.permutation(nTrSa) # Randomize the order of samples
        if (len(param) >= 4): # IF LINEAR DECAY OF THE LEARNING RAYE
            step = param[1]*(n_epoch - i)/n_epoch
#            print step
            
        for p in range (n_partitions):
            
            end = np.min(((p+1)*n_MBSa,nTrSa))
            

            X = self.Xtrain[order[p*n_MBSa:end]]
            Xg = self.GXtrain[order[p*n_MBSa:end]]
            T = self.Ytrain[order[p*n_MBSa:end]]
            
            nSa, ndim = X.shape
            nSa, nO = T.shape

            # Obtain activations and outputs needed
            Zh = np.dot(X,self.Wh) + np.tile (self.bh, (nSa,1))
            H = self.fh(Zh)
            Zg = np.dot(Xg,self.Wg) + np.tile (self.bg, (nSa,1))
            G = self.fg(Zg)
            
            dZt = np.concatenate((self.dfh(Zh),self.dfg(Zg)), axis = 1)
            Htotal = np.concatenate((H,G), axis = 1)
            
            Zo = np.dot (Htotal,self.Wo) + np.tile (self.bo, (nSa,1))
            O = self.fo(Zo)
    
            # Obtain the sigma_k for ever output neuron and sigma_j for hidden neurons
            if (self.errFunc == "MSE"):
                dError_dOut = (O - T)
            elif (self.errFunc == "EXP"):
                dError_dOut = -T * np.exp(-T * O)
            elif (self.errFunc == "CE"):
                O = O + 1/10000000   # So that division by O is not 0
                dError_dOut = -( T/O - (1-T)/(1-O))
    
    
            Sk = dError_dOut * self.dfo(Zo)
            Sk = Sk.T
            
            ##############  BOOSTING  ##############
            if (self.D_flag == 1):   # If we have given Weights to the samples
                Sk = Sk * self.D[order[p*n_MBSa:end]].T * nTrSa;
            
#           # For detecting the Divergence      
#            print "Zo: " + str(Zo)
#            print "dError_dOut: " + str(dError_dOut)
#            print "self.dfo(Zo): " + str(self.dfo(Zo))
#            
#            print "Sk: " + str(Sk)
#            print "self.Wo: " + str(self.Wo)
#            print "dZt: " + str(dZt)
#            print "self.Wh: " + str(self.Wh)
            
            Sj = np.dot(self.Wo, Sk) * dZt.T
            Sj = Sj.T
            
            Sh = Sj[:,:self.nH]
            Sg = Sj[:,self.nH:]
            
            # Obtain derivate of the error with respect to the weights
            dErrorWo = np.dot(Htotal.T,Sk.T)  # Nh * No
            dErrorWh = np.dot(X.T,Sh)  # Ni * Nh
            dErrorWg = np.dot(Xg.T,Sg)  # Ni * Nh
            
            ###### MODIFY WEIGHTS AND BIAS ########## 
            self.Wo -=  dErrorWo * step;
            self.Wh -=  dErrorWh * step;
            self.Wg -=  dErrorWg * step;
            
            self.bo -=  np.sum(Sk) * step
            self.bh -=  np.sum(Sh) * step
            self.bg -=  np.sum(Sg) * step
                
        ######### OBTAIN ERROR FOR STOPPING CRITERIA ######
        
        stop_crit = 1
        
        if ((stop_crit & PLOT_DATA) == 1):
            O = self.soft_out(self.Xtrain)
            Oval = self.soft_out(self.Xval)
            
            scoreTr[i] = self.instant_score(O, self.Ytrain)
            scoreVal[i] = self.instant_score(Oval, self.Yval)
    
            errorTr[i] = self.soft_error(O, self.Ytrain)
            errorVal[i] = self.soft_error(Oval, self.Yval)
            
            ##############  STOPPING CRITERIA  ############## 
#            stop = self.evaluate_stop(i, 1 - scoreTr, 1 - scoreVal)
#            
#            if (stop == 1):
#                n_epoch = i;
#                scoreTr = scoreTr[:n_epoch]
#                scoreVal = scoreVal[:n_epoch]
#        
#                errorTr = errorTr[:n_epoch]
#                errorVal = errorVal[:n_epoch]
#                break
            
    if (PLOT_DATA == 1):
        self.manage_results(n_epoch, scoreTr,scoreVal,errorTr,errorVal)
    