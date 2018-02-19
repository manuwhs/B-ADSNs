# -*- coding: utf-8 -*-
"""
Created on Tue Mar 24 01:31:58 2015

@author: montoya
"""
from sklearn.cross_validation import StratifiedKFold  # For crossvalidation
import numpy as np
import matplotlib.pyplot as plt

def BP_train (self, param = [0,1]):
    # Training using the Backpropagation Algorithm 
    # D is the distribution over the samples
    PLOT_DATA = self.visual[0]

    n_epoch = param[0]
    step = param[1]
    
    if (len(param) > 2):
        momentum = param[2]
        prevDeltaWo = 0
        prevDeltabo = 0
        prevDeltaWh = 0
        prevDeltabh = 0

    else:
        momentum = 0;
        
    scoreTr = np.zeros((n_epoch,1))
    scoreVal = np.zeros((n_epoch,1))

    errorTr = np.zeros((n_epoch,1))
    errorVal = np.zeros((n_epoch,1))
    
    nTrSa, Ndim = self.Xtrain.shape
        
    for i in range (n_epoch): # For every epoch
        if (PLOT_DATA == 1):
            print "Epoch: "+str(i)+"/"+str(n_epoch)
            
        order = np.random.permutation(self.nTrSa) # Randomize the order of samples
        
        for j in range (self.nTrSa):  # For every training sample
            
            # Obtain the sample to train with
            x = self.Xtrain[order[j]].reshape((1,self.nI))
            t = self.Ytrain[order[j]]

            # Obtain activations and outputs needed
            zh = np.dot(x,self.Wh) + self.bh
            h = self.fh(zh)
            zo = np.dot (h,self.Wo) + self.bo
            o = self.fo(zo)

            # Obtain derivative of error depending on the cost function            
            if (self.errFunc == "MSE"):
                dError_dOut = (o - t)
            elif (self.errFunc == "EXP"):
                dError_dOut = -t * np.exp(-t * o)
            elif (self.errFunc == "CE"):
                o = o + 1/10000000   # So that division by O is not 0
                dError_dOut = -( t/o - (1-t)/(1-o))
                
            # Obtain the sigma_k for ever output neuron and sigma_j for hidden neurons
            Sk =  dError_dOut * self.dfo(zo).reshape((self.nO,1))
            Sj = np.dot(self.Wo, Sk) * self.dfh (zh).reshape((self.nH,1))
            
            # Obtain derivate of the error with respect to the weights
            dErrorWo = np.dot(h.reshape((self.nH,1)),Sk.T)  # Nh * No
            dErrorWh = np.dot(x.T,Sj.T)  # Ni * Nh

            ##############  WEIGHTED SAMPLES  ##############
            if (self.D_flag == 1):   # If we have given Weights to the samples
                step = step * self.D[order[j]]*nTrSa;
                
            deltaWo = - dErrorWo * step;
            deltaWh = - dErrorWh * step;
            deltabo = - Sk.T * step;
            deltabh = - Sj.T * step;

            ############## REGULARIZATION ##########
            dRegulWh = 0;
            dRegulWo = 0;
            
            if (self.regularization.Reg == "L2"):
                landa = self.regularization.param[0]
                dRegulWh = self.Wh * (landa/self.nTrSa)
                dRegulWo = self.Wo * (landa/self.nTrSa)
                
            if (self.regularization.Reg == "L1"):
                landa = self.regularization.param[0]
                dRegulWh = 1 * (landa/self.nTrSa)
                dRegulWo = 1 * (landa/self.nTrSa)  
                
            deltaWo += - dRegulWo;
            deltaWh += - dRegulWh;
            
            ###############  MOMENTUM  ################

            if (momentum != 0):

                deltaWo +=  prevDeltaWo * momentum
                deltabo +=  prevDeltabo * momentum
                
                deltaWh +=  prevDeltaWh * momentum   
                deltabh +=  prevDeltabh * momentum

                prevDeltaWo = deltaWo
                prevDeltabo = deltabo
                prevDeltaWh = prevDeltaWh
                prevDeltabh = deltabh
           
            ####### MODIFY WEIGHT AND BIAS  ########
            self.Wo +=  deltaWo
            self.bo +=  deltabo
            
            self.Wh +=  deltaWh    
            self.bh +=  deltabh

        ######### OBTAIN ERROR FOR STOPPING CRITERIA ######
        Oval = self.soft_out(self.Xval)
        O = self.soft_out(self.Xtrain)
        
        scoreTr[i] = self.instant_score(O, self.Ytrain)
        scoreVal[i] = self.instant_score(Oval, self.Yval)

        errorTr[i] = self.soft_error(O, self.Ytrain)
        errorVal[i] = self.soft_error(Oval, self.Yval)
        
         ##############  STOPPING CRITERIA  ############## 
        stop = self.evaluate_stop(i, 1 - scoreTr, 1 - scoreVal)
        
        if (stop == 1):
            n_epoch = i;
            scoreTr = scoreTr[:n_epoch]
            scoreVal = scoreVal[:n_epoch]
    
            errorTr = errorTr[:n_epoch]
            errorVal = errorVal[:n_epoch]
            break
        
    if (PLOT_DATA == 1):
        self.manage_results(n_epoch, scoreTr,scoreVal,errorTr,errorVal)

        
def BP_validate (self, nH, n_iter = 10):
    # nH is the the list of nH values to validate
    # Validating values shoudld have been given.

    nParam = len(nH)
    scoreTr = np.zeros((nParam,1))
    scoreVal = np.zeros((nParam,1))
    
    for i in range (nParam):       # For every possible value of nH
        for j  in range (n_iter):  # Average over a number of tries
            self.set_nH(nH[i])   # Set new number of neurons
            self.init_Weights (distr = "uniform", param = [0,1]) # Reintilize weights
            self.ELM_train();    # Train the net
            scoreTr[i] += self.score(self.Xtrain, self.Ytrain)
            scoreVal[i] += self.score(self.Xval, self.Yval)
        scoreTr[i] /= n_iter
        scoreVal[i] /= n_iter
    
    best_indx = np.argmax(scoreVal, axis = 0 )
    best_nH = nH[best_indx]
    
    plt.figure()
    plt.plot(nH,scoreTr)
    plt.plot(nH,scoreVal)
    return (best_nH, scoreTr, scoreVal)
        
        