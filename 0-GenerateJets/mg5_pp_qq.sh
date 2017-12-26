#!/bin/sh

RUN="0"

echo "Running PP->QQ"
NAME="mg5_pp_qq_default_$RUN"
PROCESS="p p > q q"
# Could be "add process p p > mu+ mu- g g" for instance
ADDITIONAL_PROCESS=
# Could be "MG5/run_card_jj.dat" for instance. By default, only the CMS card is added
ADDITIONAL_CARDS="MG5/run_card_jj.dat"

CMD="define q=u d s u~ d~ s~
generate $PROCESS
$ADDITIONAL_PROCESS
output MG5/$NAME
launch
shower=PYTHIA8
detector=DELPHES
done
Cards/delphes_card_CMS.tcl
$ADDITIONAL_CARDS
"

echo "$CMD" | singularity run /home/iwatson/Images/CCMadgraph.img
mv MG5/$NAME/Events/run_01/tag_1_delphes_events.root root/$NAME.root
rm -rf MG5/$NAME
