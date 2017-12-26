#!/bin/sh

# Run using:
# singularity exec $HOME/Images/Madgraph.img ./run_once.sh

export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/code/MG5_aMC_v2_6_0/Delphes

./analysis_lite $@
