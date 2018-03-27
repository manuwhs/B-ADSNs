# -*- coding: utf-8 -*-
"""
Created on Tue Mar 24 01:31:58 2015

@author: montoya
"""

import numpy as np
import matplotlib.pyplot as plt
from sklearn import preprocessing

import scipy.io

# Import own libraries

import GMDH 
import pandas

plt.close('all')


#==============================================================================
# data = pandas.read_csv('../dataR/winequality-red.csv', sep = ';')
# Nsamples, Ndim = data.shape   # Get the number of bits and attr
# data_np = np.array(data, dtype = float).reshape(Nsamples, Ndim)
# X = data_np[:,:-1]
# Y = data_np[:,-1].reshape((Nsamples,1))
# Nsamples, Ndim = X.shape   # Get the number of bits and attr
# 
# #################################################################
# #################### DATA PREPROCESSING #########################
# #################################################################
# ## Split data in training and testing
# train_ratio = 0.8
# rang = np.arange(np.shape(X)[0],dtype=int) # Create array of index
# np.random.seed(0)
# rang = np.random.permutation(rang)        # Randomize the array of index
# 
# Ntrain = round(train_ratio*np.shape(X)[0])    # Number of samples used for training
# Ntest = len(rang)-Ntrain                  # Number of samples used for testing
# 
# Xtrain = X[rang[:Ntrain]]
# Xtest = X[rang[Ntrain:]]
# 
# Ytrain = Y[rang[:Ntrain]]
# Ytest = Y[rang[Ntrain:]]
#==============================================================================


mat = scipy.io.loadmat('../dataR/Airlines.mat')
Xtrain = mat["Xtrain"]
Ytrain = mat["Ytrain"]
Xtest = mat["Xtest"]
Ytest = mat["Ytest"]

#%% Normalize data
scaler = preprocessing.StandardScaler().fit(Xtrain)
Xtrain = scaler.transform(Xtrain)            
Xtest = scaler.transform(Xtest)       
        
#################################################################
#################### Neural Net Using #########################
#################################################################

Niter = 8;
M = 36;

#mySLFN = SLFN.CSLFN (nH = nH, fh = "sigmoid", fo = "sigmoid", errFunc = "CE")
myGMDH = GMDH.CGMDH (Niter = Niter,
                     M = M ); 

myGMDH.fit(Xtrain, Ytrain)                         # Train the algorithm

print "Final"
score = myGMDH.score(Xtrain, Ytrain)
print "Training Score: " + str(score)

score = myGMDH.score(Xtest, Ytest)
print "Test Score: " + str(score)


#### BARRIDO DE PARAMETROS
#trials = range (1,7)
#
#MSEtr = np.zeros((len(trials),1))
#MSEval = np.zeros((len(trials),1))
#
#i = 0
#for Niter in trials:
#    myGMDH = GMDH.CGMDH (Niter = Niter, M = M ); 
#    myGMDH.fit(Xtrain, Ytrain)                         # Train the algorithm
#    MSEtr[i] = myGMDH.score(Xtrain, Ytrain)
#    MSEval[i] = myGMDH.score(Xtest, Ytest)
#    i += 1

#plt.figure()
#plt.plot(trials,MSEtr, lw=3)
#plt.plot(trials,MSEval, lw=3)
#plt.title('MSE')
#plt.xlabel('Niter')
#plt.ylabel('MSE error')
#plt.legend(['Train','Test'])
#plt.grid()
#plt.show()
    

Otrain = myGMDH.predict(Xtrain)
Otest = myGMDH.predict(Xtest)

plt.figure()
plt.scatter(Otrain,Ytrain, lw=3)
plt.title('Regression')
plt.xlabel('Real')
plt.ylabel('Predicted')
plt.legend(['Train'])
plt.grid()
plt.show()

plt.figure()
plt.scatter(Otest,Ytest, lw=3)
plt.title('Regression')
plt.xlabel('Real')
plt.ylabel('Predicted')
plt.legend(['Test'])
plt.grid()
plt.show()