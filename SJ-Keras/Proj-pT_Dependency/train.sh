for MIN_PT in $(seq 100 100 900)
do
    MAX_PT=$(expr ${MIN_PT} + 100)
    python training.py --datasets_dir="../../Data/pt_${MIN_PT}_${MAX_PT}/3-JetImage" \
                       --log_dir="./logs/pt_${MIN_PT}_${MAX_PT}" \
                       --num_epochs=10 \
                       --kernel_size=5
done
