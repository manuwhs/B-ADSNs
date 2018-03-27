# -*- coding: utf-8 -*-
"""
Created on Tue Mar 24 01:31:58 2015

@author: montoya
"""

import numpy as np
import matplotlib.pyplot as plt
import copy as copy
class CBagging:
    
    def __init__(self, classifier = [], nB = 5, patches = "RealAdaBoost"):
        # nB: Number of baggers
        self.nB = nB
        self.set_Classifier (classifier)
        self.alphas = np.ones((nB,1))
        
    def set_Classifier (self, classifier):
        # This functions sets the classifier.
        # This classifier must have all its personal parameters set and the methods:
        # classifier.fit(Xtrain, Ytrain, W)
        # classifier.soft_out(X)
        self.base_learner = classifier
        self.learners = [];             # List of the base learners
        
    def set_Train (self, Xtrain, Ytrain):
        # Xtrain = M (N_samples, Ndim)
        # Ytrain = M (N_samples, Noutput)
        # Ytrain is expected to be in the form  Ytrain_i = [-1 -1 ··· 1 ··· -1 -1] for multiclass
        # This function MUST be called, ALWAYS
    
        self.Xtrain = Xtrain
        self.Ytrain = Ytrain
        
        shape = Xtrain.shape 
        self.nI = shape[1]      # Number of input dimensions
        self.nTrSa = shape[0]   # Number of training samples
        
        shape = Ytrain.shape 
        self.nO = shape[1]      # Number of output neurons 
        
    def set_Val (self, Xval, Yval):
        self.Xval = Xval
        self.Yval = Yval
    
    def get_alphas (self):
        
        Nsamples, nI = self.Xtrain.shape
        output = np.zeros((Nsamples,self.nB))
    
        for b in range (self.nB):
            output[:,b] = self.learners[b].get_O(self.Xtrain).flatten()
        
        Oinv = np.linalg.pinv(output)         # Get the inverse of the matrix 
        self.alphas = np.dot(Oinv,self.Ytrain)  # Get output weights
      
        
    def train (self):
        nB = self.nB
        Xtrain = self.Xtrain
        Ytrain = self.Ytrain
        
        Xval = self.Xval
        Yval = self.Yval
        
        Nsamples, Ndim = Xtrain.shape;
        NsamplesVal, Ndim = Xval.shape;
        
        Fx = np.zeros((Nsamples,1));  # Output of the whole system for every sample

        FxVal = np.zeros((NsamplesVal,1));  # Output for validation (faster than calling .score() all the time)
        
        scoreTr = np.zeros((nB,1))
        scoreVal = np.zeros((nB,1))

        for b in range (nB):  # For every weak learner
            print 'Bagger: '+ str(b) + "/"+str(nB)
            
            # Train weak learner
            learner = copy.deepcopy(self.base_learner)
            learner.init_Weights()  
            learner.train();
            
            # Updating and computing classifier output on training samples
            fm = learner.soft_out(Xtrain);  # Outputs of the Weak Classifier
            # Add the new learner to the structure
            self.learners.append(learner)

            Fx += fm * self.alphas[b];       # update strong classifier
            FxVal += learner.soft_out(Xval)* self.alphas[b]
            
            scoreTr[b] = self.instant_score(Fx,Ytrain)
            scoreVal[b] = self.instant_score(FxVal,Yval)
            
#        self.get_alphas()
            
        plt.figure()
        plt.plot(range(nB),scoreTr, lw=3)
        plt.plot(range(nB),scoreVal, lw=3)
#        plt.plot(range(T),Zt, lw=3)
        
        plt.title('Accuracy Boosting. nB ='+str(nB))
        plt.xlabel('t')
        plt.ylabel('Accuracy')
        plt.legend(['Train','Test'])  # , "Z"
        
        plt.grid()
        plt.show()

###########################################################################################################################
#########################################  Obtain resulting network ##########################################################
###########################################################################################################################

    def get_SLFN (self,tEnd = -1):
        if (tEnd < -0.5):
            tEnd = self.T 
            
        total_SLFN = copy.deepcopy(self.learners[0]);
        total_Wh = self.learners[0].Wh
        total_bh = self.learners[0].bh
        total_Wo = self.learners[0].Wo
        total_bo = self.learners[0].bo
        total_nh = self.learners[0].nh
        
        for t in range (1,tEnd):
            total_Wh = np.concatenate((total_Wh,self.learners[t].Wh), axis = 1)
        
        
#        print output
#        print self.alphas[t]
        return output

            

###########################################################################################################################
######################################### sklearn INTEFACE !!  ##########################################################
###########################################################################################################################

    def get_O (self, X):
        
        Nsamples, Ndim = X.shape;
        output = np.zeros((Nsamples,1))
        
        for b in range (self.nB):
            output += self.learners[b].get_O(X) * self.alphas[b];

#        print output
#        print self.alphas[t]
        return output
        
    def predict_proba(self,X):
        O = self.get_O(X)    # Get the output of the net
#        print O
        return O
            
    def predict(self,X):
        O = self.predict_proba(X)
        predicted = np.sign(O)
        return predicted
                
    def score(self,X,Y):
        
        predicted = self.predict(X)
        N_samples,nO = Y.shape
        score = 0.0;

        for i in range (N_samples):
#            print predicted[i], Y[i]
            if (Y[i] == predicted[i]):
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
        