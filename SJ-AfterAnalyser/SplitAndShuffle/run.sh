#!/bin/bash
IN_PATH=$1

cd $CMSSW_RELEASE_BASE
eval `scram runtime -sh`
cd -

./split_and_shuffle ${IN_PATH}
