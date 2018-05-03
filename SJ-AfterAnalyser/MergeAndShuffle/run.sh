#!/bin/bash
EVENT_TYPE=$1
DATASET_TYPE=$2

cd $CMSSW_RELEASE_BASE
eval `scram runtime -sh`
cd -

./merge_and_shuffle /cms/scratch/slowmoyang/QGJets/Data/ ${EVENT_TYPE} ${DATASET_TYPE}
