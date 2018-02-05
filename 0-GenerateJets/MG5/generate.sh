#!/bin/sh

FINAL=$1
LOW_PT=$2
HIGH_PT=$3
RUN_SUFFIX=$4
OUT_DIR=$5

echo $0
echo $1
echo $2
echo $3
echo $4

RUN="pt_${LOW_PT}_${HIGH_PT}_${RUN_SUFFIX}"

if [ $FINAL = "qq" ]
then
    PROCESS_FINAL="q q"
    CARD_SUFFIX="jj"
elif [ $FINAL = "gg" ]
then
    PROCESS_FINAL="g g"
    CARD_SUFFIX="jj"
elif [ $FINAL = "zq" ]
then
    PROCESS_FINAL="z q"
    CARD_SUFFIX="zj"
elif [ $FINAL = "zg" ]
then
    PROCESS_FINAL="z g"
    CARD_SUFFIX="zj"
else
    echo ":p"
fi


echo "Running pp->${FINAL}"
NAME="mg5_pp_${FINAL}_default_${RUN}"
PROCESS="p p > ${PROCESS_FINAL}"

ADDITIONAL_PROCESS=
ADDITIONAL_CARDS="run_card_${CARD_SUFFIX}.dat"

if [ ${CARD_SUFFIX} = "jj" ]
then
    python run_card.py ${ADDITIONAL_CARDS} \
        ptj1min=${LOW_PT}.0 ptj2min=${LOW_PT}.0 \
        ptj1max=${HIGH_PT}.0 ptj2max=${HIGH_PT}
elif [ ${CARD_SUFFIX} = "zj" ]
then
    python run_card.py ${ADDITIONAL_CARDS} \
        ptj1min=${LOW_PT}.0 ptj1max=${HIGH_PT}.0
else
    echo ":p"
fi

CMD="define q=u d s u~ d~ s~
generate $PROCESS
$ADDITIONAL_PROCESS
output $NAME
launch
shower=PYTHIA8
detector=DELPHES
done
../Cards/delphes_card_CMS.tcl
$ADDITIONAL_CARDS
"

echo "$CMD" | singularity run ~iwatson/Images/Madgraph.img
mv $NAME/Events/run_01/tag_1_delphes_events.root ${OUT_DIR}/$NAME.root
rm -rf $NAME

