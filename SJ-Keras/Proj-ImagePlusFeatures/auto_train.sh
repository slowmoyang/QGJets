START_MIN_PT=${1:-100}
END_MIN_PT=${2:-900}
for MIN_PT in $(seq ${START_MIN_PT} 100 ${END_MIN_PT})
do
    MAX_PT=$(expr ${MIN_PT} + 100)
    python training.py --datasets_dir="/home/slowmoyang/CMSSW_8_0_26/src/QGJets/Data/root_${MIN_PT}_${MAX_PT}/3-JetImage" \
                       --log_dir="./logs/root-${MIN_PT}-${MAX_PT}_$(date +%Y-%m-%d-%H-%M-%S)" \
                       --num_epochs=10 \
                       --save_freq=200 \
                       --kernel_size=5 \
                       --batch_size=512 
done
