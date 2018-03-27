import matplotlib.pyplot as plt
import numpy as np
import results_reader as rd
import manutils as mu
w = 10  # Width of the images
h = 6   # Height of the images

lw = 3  # Line weight
lw2 = 2
ls	= [ '-' , '--' , '-.' , ':' , 'steps']  # LineStyles

#########################################################################
######################## Evolution plots ##################################
#########################################################################
    
def plot_results (scoreTr,scoreVal):
        # Plots the training and Validation score of a realization,
        # In terms of the number of layers

        Nlayers = scoreTr.size
        
        plt.figure(figsize = [w,h])
        
        plt.plot(range(Nlayers),scoreTr, lw=3)
        plt.plot(range(Nlayers),scoreVal, lw=3)
        
        plt.title('Accuracy')
        plt.xlabel('Layer')
        plt.ylabel('Accuracy')
        plt.legend(['Train','Validation'])
        plt.grid()
        plt.show()

def plot_results_nH (nHs, aves,std):
        # Plots the training and Validation score of a realization

        plt.figure(figsize = [w,h])
        
        plt.plot(nHs,aves + std, lw=1, c = 'k')
        plt.plot(nHs,aves, lw=3, c = 'b')
        plt.plot(nHs,aves - std, lw=1, c = 'k')
        
        plt.title('Accuracy')
        plt.xlabel('nH')
        plt.ylabel('Accuracy')
        plt.legend(['ave + std','ave', "ave - std"])
        plt.grid()
        plt.show()
        
def plot_accu_ngamma (gammas, aves_accu, stds_accu):
        # Plots the dependance beteween the gammas window and the accuracy

        plt.figure(figsize = [w,h])
        
        plt.plot(gammas,aves_accu + stds_accu,
                 lw = lw2, c = 'k', ls = ls[1])
        plt.plot(gammas,aves_accu ,
                 lw = lw, c = 'k',  ls = ls[0])
        plt.plot(gammas,aves_accu - stds_accu, 
                 lw = lw2, c = 'k', ls = ls[1])
                 
        
        plt.title('Accuracy for different gamma window size')
        plt.xlabel('ngamma')
        plt.ylabel('Accuracy')
        plt.grid()
        plt.show()
        
def plot_nL_ngamma (gammas, aves_nLs, stds_nLs):
        
        # Plots the dependance beteween the gammas window and the number of layers required
        plt.figure(figsize = [w,h])
        
        plt.figure(figsize = [w,h])
        
        plt.plot(gammas,aves_nLs + stds_nLs,
                 lw = lw2, c = 'k', ls = ls[1])
        plt.plot(gammas,aves_nLs ,
                 lw = lw, c = 'k',  ls = ls[0])
        plt.plot(gammas,aves_nLs - stds_nLs, 
                 lw = lw2, c = 'k', ls = ls[1])
        
        plt.title('nL for different gamma window size')
        plt.xlabel('ngamma')
        plt.ylabel('Number of Layers')
        plt.grid()
        plt.show()

def plot_accu_nH (nH_list, aves_accu, stds_accu):
        # Plots the accuracy in function of the number of neurons
        # given a stop condition already

        plt.figure(figsize = [w,h])
        
        plt.plot(nH_list,aves_accu + stds_accu,
                 lw = lw2, c = 'k', ls = ls[1])
        plt.plot(nH_list,aves_accu ,
                 lw = lw, c = 'k',  ls = ls[0])
        plt.plot(nH_list,aves_accu - stds_accu, 
                 lw = lw2, c = 'k', ls = ls[1])
                 
        
        plt.title('Accuracy for different nH')
        plt.xlabel('nH')
        plt.ylabel('Accuracy')
        plt.grid()
        plt.show()
        
def plot_nL_nH (nH_list, aves_nLs, stds_nLs):
        # Plots the number of layers in function of the number of neurons
        # given a stop condition already
        
        plt.figure(figsize = [w,h])
        
        plt.figure(figsize = [w,h])
        
        plt.plot(nH_list,aves_nLs + stds_nLs,
                 lw = lw2, c = 'k', ls = ls[1])
        plt.plot(nH_list,aves_nLs ,
                 lw = lw, c = 'k',  ls = ls[0])
        plt.plot(nH_list,aves_nLs - stds_nLs, 
                 lw = lw2, c = 'k', ls = ls[1])
        
        plt.title('nL for different nH')
        plt.xlabel('nH')
        plt.ylabel('Number of layers')
        plt.grid()
        plt.show()
        

def plot_all_realizations_EVO (scoreTrs,scoreVals, new_fig = True):
        # Plots the training and Test score of all realizations of the
        # evolution across the number of layers.
        # The layer number is supossed to be the same as the number of values.

        Nruns = len(scoreTrs)  # Number of realizations
        
        if (new_fig == True):
            plt.figure(figsize = [w,h])
        
        alpha = 0.4
        for i in range(Nruns):
            Nlayers = scoreTrs[i].size
            plt.plot(range(1, Nlayers+1),scoreTrs[i], lw = lw,ls = ls[0], 
                     c = 'b', alpha = alpha)
            plt.plot(range(1, Nlayers+1),scoreVals[i], lw = lw,ls = ls[0],
                     c = 'r', alpha = alpha)
        
        plt.title('Realizations of the Evolution')
        plt.xlabel('Number of layers')
        plt.ylabel('Accuracy')
        plt.legend(["Tr", "val"],loc = 4)
        plt.grid()
        plt.show()
        
def plot_results_ave_std (aves, std, new_fig = True):
        # Plots the average and std of the sequence
        # in function of the number of layers
        Nlayers = aves.size
        
        if (new_fig == True):
            plt.figure(figsize = [w,h])
        
        plt.plot(range(1, Nlayers+1),aves + std, 
                 lw = lw2, c = 'k', ls = ls[1])
        plt.plot(range(1, Nlayers+1),aves, 
                 lw = lw, c = 'k',  ls = ls[0])
        plt.plot(range(1, Nlayers+1),aves - std, 
                 lw = lw2, c = 'k', ls = ls[1])
        
        plt.title('Accuracy')
        plt.xlabel('Number of layers')
        plt.ylabel('Accuracy')
        plt.legend(['ave + std','ave', "ave - std"])
        plt.grid()
        plt.show()

def plot_gamma_nL (aves, std, new_fig = True):
        # Plots the average and std accu of the sequence
        # in function of the number of layers
        Nlayers = aves.size
        
        if (new_fig == True):
            plt.figure(figsize = [w,h])
        
        plt.plot(range(1, Nlayers+1),aves + std, 
                 lw = lw2, c = 'k', ls = ls[1])
        plt.plot(range(1, Nlayers+1),aves, 
                 lw = lw, c = 'k',  ls = ls[0])
        plt.plot(range(1, Nlayers+1),aves - std, 
                 lw = lw2, c = 'k', ls = ls[1])
        
        plt.title('Gamma evolution')
        plt.xlabel('Number of layers')
        plt.ylabel('Gamma')
        plt.legend(['ave + std','ave', "ave - std"])
        plt.grid()
        plt.show()
        
def plot_tr_val_nL (nLs_list, aves_OMN, aves_OMN_tr, std_OMN, std_OMN_tr, new_fig = True):
        # Plots the training and Validation score +- std 
        # for a given number of epoch and Nh, in function of nL
        
        if (new_fig == True):
            plt.figure(figsize = [w,h])
        
        n_std = 1
        
        plt.plot(nLs_list,aves_OMN + n_std*std_OMN, 
                 lw=lw2, c = '0.0', ls = ls[1])
        plt.plot(nLs_list,aves_OMN, 
                 lw=lw, c = '0.0', ls = ls[0], label = "Test")
        plt.plot(nLs_list,aves_OMN - n_std*std_OMN, 
                 lw=lw2, c = '0.0', ls = ls[1])
        
        
        plt.plot(nLs_list,aves_OMN_tr + n_std*std_OMN_tr, 
                 lw=lw2, c = '0.4', ls = ls[1])
        plt.plot(nLs_list,aves_OMN_tr, 
                 lw=lw, c = '0.4',ls = ls[0], label = "Train")
        plt.plot(nLs_list,aves_OMN_tr - n_std*std_OMN_tr, 
                 lw=lw2, c = '0.4', ls = ls[1])
        
#        plt.title(r'Accuracy $\pm$ std (nL)')
        plt.xlabel('Number of learners '+r'$l$', fontsize=15)
        plt.ylabel(r'$\%$'+' accuracy rate ' +r'$\pm std$',fontsize=15)
        plt.legend(loc = 4)
        plt.grid()
        plt.show()
 
def plot_acc_nL_nH (nHs_list, aves):
        # Plots the training and Validation score of a realization
        
        # Color = [ 0 - 1] of grays
        
        plt.figure(figsize = [w,h])
        N_neurons = len(nHs_list)
#        N_neurons = len(aves)
#        print len(aves)
#        print len(nHs_list)
        ep = 0.2
#        print "GFEDDFDFB"
        for i in range (N_neurons):
            Nlayers = aves[i].size
            plt.plot(range(1, Nlayers+1),aves[i], lw=3, 
                     c = str(1-(ep + ((1-ep)*(i))/N_neurons)), 
                     label = str(nHs_list[i]))


        plt.title('Accuracy (nL, nH)')
        plt.xlabel('Number of Layers')
        plt.ylabel('Accuracy')
        plt.legend(loc = 4, fontsize = 7)
        plt.grid()
        plt.show()

def plot_3D_nH_nL (nHs,nLs, aves_OMN):
        # Plots the training and Validation score of a realization

    X = np.array(nHs)
    Y = np.array(nLs)
    X, Y = np.meshgrid(X, Y)
    
    Z = mu.convert_to_matrix(aves_OMN).T

    from mpl_toolkits.mplot3d import Axes3D
    from matplotlib import cm
    from matplotlib.ticker import LinearLocator, FormatStrFormatter
    import matplotlib.pyplot as plt
    
    fig = plt.figure()
    ax = fig.gca(projection='3d')
    
    surf = ax.plot_surface(X, Y, Z, rstride=1, cstride=1, cmap=cm.coolwarm,
                           linewidth=0, antialiased=False)
    
    ax.set_zlim(np.min(Z.flatten()), np.max(Z.flatten()))
    ax.zaxis.set_major_locator(LinearLocator(10))
    ax.zaxis.set_major_formatter(FormatStrFormatter('%.02f'))
    
    plt.xlabel('Hidden neurons')
    plt.ylabel('Number of Layers')
    fig.colorbar(surf, shrink=0.5, aspect=5)
 
    ax.set_zlabel('Z Label')
    
    plt.show()

def plot_3D_a_nH (alphas,nHs, aves_OMN):
        # Plots the training and Validation score of a realization

    X = np.array(alphas)
    Y = np.array(nHs)
    X, Y = np.meshgrid(X, Y)
    
    Z = mu.convert_to_matrix(aves_OMN).T

    from mpl_toolkits.mplot3d import Axes3D
    from matplotlib import cm
    from matplotlib.ticker import LinearLocator, FormatStrFormatter
    import matplotlib.pyplot as plt
    
    fig = plt.figure()
    ax = fig.gca(projection='3d')
    
    surf = ax.plot_surface(X, Y, Z, rstride=1, cstride=1, cmap=cm.coolwarm,
                           linewidth=0, antialiased=False)
    
    ax.set_zlim(np.min(Z.flatten()), np.max(Z.flatten()))
    ax.zaxis.set_major_locator(LinearLocator(10))
    ax.zaxis.set_major_formatter(FormatStrFormatter('%.02f'))
    
    plt.xlabel('alpha')
    plt.ylabel('Hidden neurons')
    fig.colorbar(surf, shrink=0.5, aspect=5)
 
    ax.set_zlabel('Z Label')
    
    plt.show()
    
    
def plot_3D_a_b (alphas,betas, aves_OMN):
        # Plots the training and Validation score of a realization

    X = np.array(alphas)
    Y = np.array(betas)
    X, Y = np.meshgrid(X, Y)
    
    Z = mu.convert_to_matrix(aves_OMN).T

    from mpl_toolkits.mplot3d import Axes3D
    from matplotlib import cm
    from matplotlib.ticker import LinearLocator, FormatStrFormatter
    import matplotlib.pyplot as plt
    
    fig = plt.figure()
    ax = fig.gca(projection='3d')
    
    plt.tight_layout(pad=2, w_pad=0.5, h_pad=1.0)
    
    
    surf = ax.plot_surface(X, Y, Z, rstride=1, cstride=1, cmap=cm.gray_r,
                           linewidth=0)
    
    ax.set_zlim(np.min(Z.flatten()), np.max(Z.flatten()))
    
    ax.zaxis.set_major_locator(LinearLocator(8))
    ax.zaxis.set_major_formatter(FormatStrFormatter('%.02f'))
    
    plt.xlabel(r'$\alpha$', fontsize=25)
    plt.ylabel(r'$\beta$', fontsize=25)
#    fig.colorbar(surf, shrink=0.5, aspect=5)
 
    ax.set_zlabel('% accuracy rate', fontsize=15)
    
    plt.show()
#########################################################################
######################## Final Results plots ##################################
#########################################################################

def plot_tr_val_tst_nH (nHs, aves_CV, aves_OMN,aves_OMN_tr,std_CV, std_OMN, std_OMN_tr):
        # Plots the training and Validation score of a realization
        plt.figure(figsize = [w,h])
        
        n_std = 1
        
        plt.plot(nHs,aves_OMN + n_std*std_OMN, lw=2, c = 'k', ls ='--')
        plt.plot(nHs,aves_OMN, lw=3, c = 'k', label = "TsT Acc")
        plt.plot(nHs,aves_OMN - n_std*std_OMN, lw=2, c = 'k', ls ='--')
        
        plt.plot(nHs,aves_CV + n_std*std_CV, lw=2, c = 'r', ls ='--')
        plt.plot(nHs,aves_CV, lw=3, c = 'r', label = "Val Acc")
        plt.plot(nHs,aves_CV - n_std*std_CV, lw=2, c = 'r', ls ='--')
        
        plt.plot(nHs,aves_OMN_tr + n_std*std_OMN_tr, lw=2, c = 'b', ls ='--')
        plt.plot(nHs,aves_OMN_tr, lw=3, c = 'b', label = "Tr Acc")
        plt.plot(nHs,aves_OMN_tr - n_std*std_OMN_tr, lw=2, c = 'b', ls ='--')
        
        plt.title('Accuracy of tr, val and tst')
        plt.xlabel('Hidden neurons')
        plt.ylabel('Accuracy')
        plt.legend(loc = 4)
        plt.grid()
        plt.show()
        


def plot_tr_tst_nH (nHs, aves_OMN, aves_OMN_tr, std_OMN, std_OMN_tr):
        # Plots the training and Validation score of a realization.
        # Then points are scattered.

        plt.figure(figsize = [w,h])
        
        n_std = 2
        
        plt.plot(nHs,aves_OMN + n_std*std_OMN, lw=2, c = 'k', ls ='--')
        plt.plot(nHs,aves_OMN, lw=3, c = 'k', label = "TsT Acc")
        plt.plot(nHs,aves_OMN - n_std*std_OMN, lw=2, c = 'k', ls ='--')
        
        
        plt.plot(nHs,aves_OMN_tr + n_std*std_OMN_tr, lw=2, c = 'b', ls ='--')
        plt.plot(nHs,aves_OMN_tr, lw=3, c = 'b', label = "Tr Acc")
        plt.plot(nHs,aves_OMN_tr - n_std*std_OMN_tr, lw=2, c = 'b', ls ='--')
        
        plt.title('Accuracy')
        plt.xlabel('Hidden neurons')
        plt.ylabel('Accuracy of tr and tst with distribution')
        plt.legend(loc = 4)
        plt.grid()
        plt.show()
        
    
def scatter_points (nHs, points_OMN_tr, points_OMN):
        # Plots the training and Validation score of a realization
#        plt.figure()
        num = len(nHs)
        alpha = 0.15
        for i in range (num):
            n_OMN_tr = points_OMN_tr[i].size
            n_OMN = points_OMN[i].size
            
            plt.scatter(nHs[i]*np.ones((n_OMN_tr,1)),points_OMN_tr[i], lw=1, c = 'b', alpha =alpha)
            plt.scatter(nHs[i]*np.ones((n_OMN,1)),points_OMN[i], lw=1, c = 'k', alpha = alpha)
            
 
def plot_all_OMN_nH (nH_list,Nepoch_list,tr_val_tst ,All_Exec_list_OMN):
        # Plots the different omniscient error curves for different number of epochs

        plt.figure(figsize = [w,h])
        
        N_plots = len(Nepoch_list)
        
        for i in range(N_plots):
            if (All_Exec_list_OMN[i] != []):  # If there is data
                aves_OMN, stds_OMN = rd.get_ave_and_std (All_Exec_list_OMN[i])
                
                plt.plot(nH_list,aves_OMN[:,tr_val_tst], lw=3, label = str(Nepoch_list[i]))
    
        plt.title('Accuracy')
        plt.xlabel('Hidden neurons')
        plt.ylabel('Accuracy for differen Nepoch')
        plt.legend(loc = 4)
        plt.grid()
        plt.show()
        
        
        


















































def plot_all_tr_val_nL_surface_XXXXXXXXX (nHs,nLs, aves_OMN, aves_OMN_tr):
        # Plots the training and Validation score of a realization

    N_neurons = len(aves_OMN)
    
    X = np.array(nHs)
    Y = np.array(nLs)
    X, Y = np.meshgrid(X, Y)
    
#    print X
#    print Y
    Z = mu.convert_to_matrix(aves_OMN).T
#    Z2 = mu.convert_to_matrix(aves_OMN_tr).T
#    Z = Z.flatten()
    
#    print X.shape, Y.shape
#    print Z.shape
    
    from mpl_toolkits.mplot3d import Axes3D
    from matplotlib import cm
    from matplotlib.ticker import LinearLocator, FormatStrFormatter
    import matplotlib.pyplot as plt
    
    fig = plt.figure()
    ax = fig.gca(projection='3d')
    
    """ VAL """
    surf = ax.plot_surface(X, Y, Z, rstride=1, cstride=1, cmap=cm.coolwarm,
                           linewidth=0, antialiased=False)
    
    ax.set_zlim(np.min(Z.flatten()), np.max(Z.flatten()))
    ax.zaxis.set_major_locator(LinearLocator(10))
    ax.zaxis.set_major_formatter(FormatStrFormatter('%.02f'))
    
    plt.xlabel('Hidden neurons')
    plt.ylabel('Number of Layers')
    fig.colorbar(surf, shrink=0.5, aspect=5)
 
    """ TR """
#    surf = ax.plot_surface(X, Y, Z2, rstride=1, cstride=1, cmap=cm.coolwarm,
#                           linewidth=0, antialiased=False)
#    
#    ax.set_zlim(np.min(Z2.flatten()), np.max(Z2.flatten()))
#    ax.zaxis.set_major_locator(LinearLocator(10))
#    ax.zaxis.set_major_formatter(FormatStrFormatter('%.02f'))
#    
#    plt.xlabel('Hidden neurons')
#    plt.ylabel('Number of Layers')
#    fig.colorbar(surf, shrink=0.5, aspect=5)
#    
#    plt.show()


#==============================================================================
#     X = np.array(nHs)
#     Y = np.array(nLs)
# #    X, Y = np.meshgrid(nHs, nLs)
#     
#     print X
#     print Y
#     Z = convert_to_matrix(aves_OMN)
# #    Z = Z.flatten()
#     
#     print X.shape, Y.shape
#     print Z.shape
# 
#     from mpl_toolkits.mplot3d import Axes3D
#     import matplotlib.pyplot as plt
#     
#     
#     fig = plt.figure()
#     ax = fig.add_subplot(111, projection='3d')
#     
#     for x in range(X.size):
#         for y in range(Y.size):
#             ax.scatter(X[x], Y[y], Z[x,y])
#     
#     ax.set_xlabel('X Label')
#     ax.set_ylabel('Y Label')
#==============================================================================
    ax.set_zlabel('Z Label')
    
    plt.show()
        