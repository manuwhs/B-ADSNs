#!/bin/bash
export LD_PRELOAD=/opt/intel/composerxe/mkl/lib/intel64/libmkl_mc.so:/opt/intel/composerxe/mkl/lib/intel64/libmkl_def.so:/opt/intel/composerxe/mkl/lib/intel64/libmkl_core.so:/opt/intel/composerxe/mkl/lib/intel64/libmkl_intel_thread.so:/opt/intel/composerxe/compiler/lib/intel64/libiomp5.so

python ./SLFN/main.py 20 & python ./SLFN/main.py 25 & python ./SLFN/main.py 30


