#MIN_PT=${1:-100}
#MAX_PT=$(expr ${MIN_PT} + 100)
python training.py --datasets_dir="/scatch/slowmoyang/QGJets/pt_100_200/3-JetImage/Shuffled" \
                   --log_dir="./logs/pt_100_200_$(date +%Y%m%d-%H%M%S)" \
                   --num_epochs=10 \
                   --kernel_size=5
