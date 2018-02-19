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
    SHOW_EPOCHS = self.visual[0]

    n_epoch = param[0]
    step_ini = param[1]
    Roh = float(param[2])

    nTrSa, Ndim = self.Xtrain.shape
    
    scoreTr = np.zeros((n_epoch,1))
    scoreVal = np.zeros((n_epoch,1))
    
    for i in range (n_epoch): # For every epoch
        if (PLOT_DATA == 1):
            print "Epoch: "+str(i)+"/"+str(n_epoch)
            
        step = step_ini*(n_epoch - i)/n_epoch
        order = np.random.permutation(nTrSa) # Randomize the order of samples
#        print step
     
        for j in range (nTrSa):  # For every training sample
            x = self.Xtrain[order[j]].reshape((1,self.nI))
            xg = self.GXtrain[order[j]].reshape((1,self.nIG))  # God's part
            t = self.Ytrain[order[j]]

            # Obtain activations and outputs needed
            zh = np.dot(x,self.Wh) + self.bh
            h = self.fh(zh)
            zg = np.dot(xg,self.Wg) + self.bg
            g = self.fg(zg)
            
            htotal = np.concatenate((h,g),axis = 1)
        
            zo = np.dot (htotal,self.Wo) + self.bo
            o = self.fo(zo)

            # Obtain the sigma_k for ever output neuron and sigma_j for hidden neurons
            if (self.errFunc == "MSE"):
                dError_dOut = (o - t)
                
            Sk =  dError_dOut * self.dfo(zo).reshape((self.nO,1))
            Sj = np.dot(self.Wo, Sk) * np.concatenate((self.dfh (zh).reshape((self.nH,1)),
                                                      self.dfg(zg).reshape((self.nG,1))), axis = 0)
            Sh = Sj[:self.nH]
            Sg = Sj[self.nH:]
            
            # Obtain derivate of the error with respect to the weights
            dErrorWo = np.dot(htotal.reshape((self.nH + self.nG,1)),Sk.T)  # Nh * No
            dErrorWh = np.dot(x.T,Sh.T)  # Ni * Nh
            
            dErrorWg = np.dot(xg.T,Sg.T)  # Input weights of the G neurons neurons
            
            ##############  WEIGHTED SAMPLES  ##############
            if (self.D_flag == 1):   # If we have given Weights to the samples
                step = step * self.D[order[j]]*nTrSa; # The nTrSa is undo the normalization
            
            step_o = step/Roh   # Anibals says to lower the output constant
            # Modify the weights and bias
            self.Wo -=  dErrorWo * step;
            
            """ DONT LET THE Wo of the G neuron to be trained""" 
            self.Wo[self.nH:,:] = np.ones((1, self.nO))*self.ro_G
#            print self.ro
            
            self.Wh -=  dErrorWh * step;
#            self.Wg -=  dErrorWg * step;
            
            self.bo -=  Sk.T * step_o
            self.bh -=  Sh.T * step_o
#            self.bg -=  Sg.T * step_o
            
#        scoreTr[i] = self.score(self.Xtrain,self.GXtrain, self.Ytrain)
#        scoreVal[i] = self.score(self.Xval,self.GXval, self.Yval)
        
    if (PLOT_DATA == 1):
        plt.figure()
        plt.plot(range(n_epoch),scoreTr, lw=3)
        plt.plot(range(n_epoch),scoreVal, lw=3)
        plt.title('Accuracy BP. step ='+str(step))
        plt.xlabel('Epoch')
        plt.ylabel('Accuracy')
        plt.legend(['Train','Test'])
        plt.grid()
        plt.show()
        
    
        
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
        
        