
import numpy as np
#################  PARAMETROS DEL BARRIDO ##################
## Here we introduce the init and end values of the cluster
## These are constant
""" POR convenio, cuando alpha = 1, beta = 5 para lectura y tenemos que
hacer que por dentro se establezcan mecanismos para copiar para la representacion. """
fo_list = [1]
Learning_Rate_list = [0]  # 2 for database 1 y 3
BatchSize_list = [0]
Inyection_list =  [2]
Enphasis_list = [3]
CV_List = [0]     # 1 is for training with all dataset
Nruns_List = [0]

Ninit_list = [1,4,8,16]
Ninit_indx_list = [0,1,2,3]

Roh_list = [1,2,4,8]
Roh_indx_list = [1,2,3]

### These are the ones we actually search
nH_list = [2,3,4,5,6,7,8,9,10,11,
           12,13,14,15,16,17,18,19,20,21,
           22,23,24,25,26,27,28,29,30,31,
           32,33,34,35,36,37,38,39,40,41,
           42,43,44,45,46,47,48,49,50]

#nH_indx_list = [8,10,12,14]    #  1
#nH_indx_list = [3,8,13,18,23,28,33,38]    #  2
#nH_indx_list = [1,2,3,4,5,6,7]             #  4   
#nH_indx_list = [1,2,3,4,5,6]             #  7  
#nH_indx_list = [3,8,13,18,23,28,33]    #  5
#nH_indx_list = [8,12,16,20,24,28,32,36,40,44,48]    #  3
#nH_indx_list = [8,12,16,20,24,28]    #  3
nH_indx_list = [29,33,37]    #  3_2

#nH_indx_list = [2]         
#nH_indx_list = [0,2,4,6,8,10,12,14,16,18,20,22,24,26,28,30,32,34,36,38,40,42,44,46]
#nH_indx_list = [8,12,16,20,24,28,32,36,40,44,48]

#N_epochs_indx_list = [2]     # 1
#N_epochs_indx_list = [2,3,4]   # 2
#N_epochs_indx_list = [1,2,3]  # 4
#N_epochs_indx_list = [0,1,2,3]  # 7
#N_epochs_indx_list = [3,4,5]  # 5
#N_epochs_indx_list = [1,2,3]  # 7
#N_epochs_indx_list = [1,2,3]  # 3
N_epochs_indx_list = [2,3]  # 3_2

N_epochs_list = [25,50,100,150,200,300]



#N_epochs_list = [25,50,200]
#N_epochs_indx_list = [0,1,2,3]

alpha_list = [0,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1.0]
alpha_indx_list = [0,1,2,3,4,5,6,7,8,9,10]
#alpha_indx_list = [0,7,8,9]
#alpha_indx_list = [7,8,9]
#alpha_indx_list = [0,2,4,6,8,10]
#alpha_indx_list = [10]

#alpha_list = [0.01,0.02,0.03,0.05,0.07]
#alpha_indx_list = [0,1,2,3,4]
#
beta_list = [0,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1]
beta_indx_list = [0,1,2,3,4,5,6,7,8,9,10]
#beta_indx_list = [6,7,8,9,10]
#beta_indx_list = [0,2,4,6,8,10]
#beta_indx_list = [5,6,7,8,9,10]
#beta_indx_list = [0]
#beta_list = [0,0.5,1.0]
#beta_indx_list = [0,5,10]

#beta_list = [0.5]
#beta_indx_list = [5]



""" FOR TRAINING """

#nH_list = [8,10,12,14,16]
#nH_list = [2,3,4,5,8]
#nH_list = [20,25,30,40]
#nH_list = [25,30,35,40]
#nH_indx_list = [0,1,2,3,4]


#alpha_list = [0.8,0.9,1]
#alpha_indx_list = [8,9,10]

#alpha_list = [0.1,0.2]
#alpha_indx_list = [1,2]

#alpha_list = [0.5,0.7]
#alpha_indx_list = [5,7]

#beta_list = [0,0.5,1.0]
#beta_indx_list = [0,5,10]

#beta_list = [0.5]
#beta_indx_list = [5]


class PPE_params:
    
    def __init__(self):

        self.fo_list = fo_list 
        self.Learning_Rate_list = Learning_Rate_list 
        self.BatchSize_list = BatchSize_list 
        self.Inyection_list = Inyection_list 
        self.Enphasis_list = Enphasis_list
        self.CV_List = CV_List 
        self.Nruns_List = Nruns_List 

        self.Ninit_list = Ninit_list
        self.Ninit_indx_list = Ninit_indx_list
        
        self.Roh_list = Roh_list
        self.Roh_indx_list = Roh_indx_list
        ### These are the ones we actually search
        self.nH_list = nH_list 
        self.nH_indx_list = nH_indx_list 
        
        self.N_epochs_list = N_epochs_list 
        self.N_epochs_indx_list = N_epochs_indx_list 
        
        self.alpha_list = alpha_list 
        self.alpha_indx_list = alpha_indx_list 
        
        self.beta_list = beta_list 
        self.beta_indx_list = beta_indx_list 
        
PPE_p = PPE_params()

PPE_p.nH_list = np.array(PPE_p.nH_list)  # Esta conversion es para luego poderle pasar una lista de indices
PPE_p.N_epochs_list = np.array(PPE_p.N_epochs_list)
PPE_p.alpha_list = np.array(PPE_p.alpha_list)
PPE_p.beta_list = np.array(PPE_p.beta_list)