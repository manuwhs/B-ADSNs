# -*- coding: utf-8 -*-
"""
Created on Tue Mar 24 01:31:58 2015

@author: montoya
"""

import numpy as np


#########################################
""" OUTPUT FUNCTION OBTAINING """
#########################################

def get_O (self, X, lEnd = -1):
    # Gets the output of the system for the first lEnd layers
    if (lEnd < - 0.5):
        lEnd = self.nL 
#        print lEnd
    
    Nsamples, Ndim = X.shape;
    
    PrevZos = [];         # Previous outputs of the system for training
    Xlayer = X          # Concatenation of X and previous outputs (X of the layer)
      
    for l in range (lEnd):  # For every layer !
  
       # Set Previous outputs for the next layer
  

        if (l == 0): 
            LastZo = self.fi(self.layers[l].get_Zo(Xlayer))
            PrevZos = LastZo
        else:
            
            Xlayer = np.concatenate((X,PrevZos), axis = 1)
            LastZo = self.fi(self.layers[l].get_Zo(Xlayer))
            output = self.layers[l].get_O(Xlayer);  # Get output of this layer.
            """ get_Zo or get_O """
#            print PrevZos.shape
            PrevZos = np.concatenate((PrevZos, LastZo), axis = 1)

            if (l >= self.nP):   # IF we have reached the total output inyection number
                PrevZos = np.delete(PrevZos, 0, axis = 1)
        
        
      
    """ NOTICE WE PROPAGATE THE ACTIVATION """
      
#    output = self.layers[l].fo(output)
#        print output
    return output
    
    
    
    
    