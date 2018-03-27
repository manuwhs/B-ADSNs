# -*- coding: utf-8 -*-
"""
Created on Tue Mar 24 01:31:58 2015

@author: montoya
"""

import numpy as np
import matplotlib.pyplot as plt
import copy as copy
import time
import paramClasses as paC


def train_once (self):  # The D is actually not needed in boosting case since it is intialized to W = uniform
    
    T = self.Tmax
    
    Xtrain = self.Xtrain
    Ytrain = self.Ytrain
    
    Xval = self.Xval
    Yval = self.Yval
    
    Nsamples, Ndim = Xtrain.shape;
    NsamplesVal, Ndim = Xval.shape;
    
    self.alphas = np.ones((T,1))    
    
    W  = np.ones((Nsamples,1));   # Weight of each sample

    Fx = np.zeros((Nsamples,1));  # Output of the whole system for every sample
    FxVal = np.zeros((NsamplesVal,1));  # Output for validation (faster than calling .score() all the time)
    
    scoreTr = np.zeros((T,1))
    scoreVal = np.zeros((T,1))
    Zt =  np.ones((T,1))   # Value of the normalization constant
    Zt[0] = 1;
    
    self.learners = []  # Empty learners
    
    for t in range (T):  # For every weak learner
        
        W = W / np.sum(W);  # Normalize weights
#        print 'Boosting Round: '+ str(t+1) + "/"+str(T)
#        print W
        # Train weak learner
        learner = copy.deepcopy(self.base_learner)
        
        learner.set_Train(self.Xtrain, self.Ytrain)
        learner.set_Val(self.Xval, self.Yval)
        
        learner.init_Weights()  
        
#        print "Training"
        learner.set_D(W);   # Set samples distribution
        learner.train_once(); # SOLO UN PUTO ENTRENAMIENTO
        
        ###########################################################
        # Updating and computing classifier output on training samples
        ###########################################################
        
        fm = learner.soft_out(Xtrain);  # Outputs of the Weak Classifier

        # Obtain the dehenphasis alpha of the learner given by RealAdaboost
        
        if (self.alg == "GentleBoost"):
             self.alphas[t] = 1;    # For the GentleBoost
             
        elif (self.alg == "RealAdaBoost"):
            r = np.sum(W * Ytrain * fm);
            self.alphas[t] = np.log((1+r)/(1-r))/2;
        
        Fx += fm * self.alphas[t];       # update strong classifier
        FxVal += learner.soft_out(Xval)* self.alphas[t]
        
        
        # Reweight training samples    
        W = W * np.exp(-Ytrain * fm * self.alphas[t]);
        
        # Add the new learner to the structure
        self.learners.append(learner)
        
        # Obtain the normalization constant (error upper bound)
        Zt[t] = Zt[t - 1] * np.sum(W)
        
        scoreTr[t] = self.instant_score(Fx,Ytrain)
        scoreVal[t] = self.instant_score(FxVal,Yval)
        
        print "Alpha: " + str(self.alphas[t])
        print "Score Tr: "+ str(scoreTr[t]) + " Score Val: " + str(scoreVal[t]) 
    
    ############ SELECT THE VALDATED BEST SOLUTION ##################
#    best_T = np.argmin(scoreVal)
#    self.T = best_T + 1
#    
#    print self.T
    
#    print scoreTr[t], scoreVal[t], self.alphas[t]
        
    plt.figure()
    plt.plot(range(T),scoreTr, lw=3)
    plt.plot(range(T),scoreVal, lw=3)
#        plt.plot(range(T),Zt, lw=3)
    
    plt.title('Accuracy Boosting. T ='+str(T))
    plt.xlabel('t')
    plt.ylabel('Accuracy')
    plt.legend(['Train','Test'])  # , "Z"
    
    plt.grid()
    plt.show()
        
        
        



def train (self):  # SAME AS THE SLFN but showing intermediate results
 
    # Adapt the labels so that they are correct (-1 or 0 and transform multivariate if needed)
    self.Ytrain = paC.adapt_labels(self.Ytrain, mode = self.outMode)
    self.Yval = paC.adapt_labels(self.Yval, mode = self.outMode )
    if (self.Xtest != []):     # If there is a test dataset.
        self.Ytest = paC.adapt_labels(self.Ytest, mode = self.outMode )
        
    self.TrError = np.zeros((self.Nruns,1))
    self.ValError = np.zeros((self.Nruns,1))
    self.TstError = np.zeros((self.Nruns,1))
    
    for r in range (self.Nruns):
        self.train_CV(r = r)
        print self.TrError[r], self.ValError[r] , self.TstError[r]
