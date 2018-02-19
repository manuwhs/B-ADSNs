#!/bin/bash
# Load li
unset LD_LIBRARY_PATH
# printenv

export LD_PRELOAD=/opt/intel/composerxe/mkl/lib/intel64/libmkl_mc.so:/opt/intel/composerxe/mkl/lib/intel64/libmkl_def.so:/opt/intel/composerxe/mkl/lib/intel64/libmkl_core.so:/opt/intel/composerxe/mkl/lib/intel64/libmkl_intel_thread.so:/opt/intel/composerxe/compiler/lib/intel64/libiomp5.so

python ./GDSNBoost/main_cluster3.py $* >stdout.log 2>stderr.log

# The main.py is the file to execute. All executing files should be in the /commonFiles folder
# The datasets and output should be in 
# /export/clusterdata/mmontoya/data
# /export/clusterdata/mmontoya/Results
# We will reference these folders with absolute paths. Y a tomar por culo !!

# The $* is the enviroment variables sent that will be introduced by the cluster
# using the configured GridTask.

# stdout.log will save all the print of the program
# stderr.log will save all the errors of the program
