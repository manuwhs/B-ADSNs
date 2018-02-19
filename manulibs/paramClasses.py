
import numpy as np

class Init_Distrib():
    # Class that contains the parameters for initialing the weights in a general form
    def __init__(self, weightInit = "default", init_param = ['uniform',-1,1]):
        self.weightInit = weightInit;  # Initialization distribution
        self.param = init_param        # Paramters for the initialization
 
class Training_Alg():
    # Class that contains the parameters of the training algorithm.
    # Each algorithm is defined by its name and a set of parameters.
    def __init__(self, trAlg = "BP", trAlgParams = []):
        self.trAlg = trAlg;                 # Initialization distribution
        self.param = trAlgParams        # Paramters for the initialization

class Regularization():
    # Class that contains the paramters and type of regularization:
    # L2
    # L1
    # Dropout
    # Generating 

    def __init__(self, Reg = "L2", RegParams = [0.1]):
        self.Reg = Reg;                 # Initialization distribution
        self.param = RegParams        # Paramters for the initialization

class Init_Centers():
    # Class that contains the parameters for initialing the weights in a general form
    def __init__(self, centersInit = "randomSamples", init_param = []):
        self.centersInit = centersInit;  # Initialization distribution
        self.param = init_param        # Paramters for the initialization

class Stop_Criterion():
    def __init__(self, StopCrit = "Nmax", Stop_params = []):
        self.StopCrit = StopCrit;  # Initialization distribution
        self.param = Stop_params        # Paramters for the initialization
        












def get_labels(Y, mode = 0):
    # It accepts a vector Y of labels [1, 3, 5,5,3,6...]
    # It outputs a matrix Ycod with the decomposed version in 0 1 or -1 1
    Nsam = np.alen(Y)
    nameClasses = np.unique(Y)   # Real name of the classes
    nClasses = np.alen(nameClasses)   # number of classes

    Ycod = mode * np.ones((Nsam,nClasses))
    
    # Set the corresponding ones
    for i in range(Nsam):
        ind = np.where(nameClasses == Y[i])
        Ycod[i ,ind] = 1
        
    return Ycod

def adapt_labels(Y, mode = 0):
    # Convert the Y vector to 0 - 1 or -1 1
    # CONVERT THIS FUNCTION TO MAKE IT USABLE FOR BOTH ARRAYS AND MATRIXES
    shape = Y.shape
    Nsam = len(Y)
    
#    print "RGRGR"
#    print Y
#    print Y.shape
    
    Y = Y.flatten()
    # Set the corresponding ones
    for i in range(Nsam):
        if (Y[i] != 1):
            Y[i] = mode
    Y = Y.reshape(shape)
    return Y