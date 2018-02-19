import pickle
import gc
import os
def store_pickle (filename, li, partitions = 1, verbose = 0):
    gc.collect()
    # This function stores the list li into a number of files equal to "partitions" in pickle format
    num = int(len(li)/partitions);
    
    for i in range(partitions - 1):
        
        if (verbose == 1):
            print "Creating file: " + filename + str(i)+ ".pkl"
            
        with open(filename + str(i)+ ".pkl", 'wb') as f:
            pickle.dump(li[i*num:(i+1)*num], f)    
            # We dump only a subset of the list
            
    if (verbose == 1):
        print "Creating file: " + filename + str(partitions -1)+ ".pkl"
    
    with open(filename + str(partitions - 1)+ ".pkl", 'wb') as f:
            pickle.dump(li[num*(partitions - 1):], f)    
            # We dump the last subset.
    gc.collect()
    
def load_pickle (filename, partitions = 1, verbose = 1):
    gc.collect()
    total_list = []
    for i in range(partitions):
        if (verbose == 1):
            print "Loading file: " + filename + str(i)+ ".pkl"
        
        if (os.path.exists(filename + str(i)+ ".pkl") == True):   # Check if file exists !!

            with open(filename + str(i)+ ".pkl", 'rb') as f:
                part = pickle.load(f)    # We dump the auctions one by one coz pickle uses a lot of memmory
            total_list.extend(part)
        
        else:
            print "File does not exist: " + filename + str(i)+ ".pkl"
            return []
            
    gc.collect()
    return total_list

#==============================================================================
# n = 3
# lista = [10, 23, 43, 65, 34, 98, 90, 84, 98]
# 
# store_pickle("lista",lista,n)
# 
# lista2 = load_pickle("lista",n)
#==============================================================================


