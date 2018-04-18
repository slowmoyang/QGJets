LOG_NAME=${1:-Run-$(date +"%Y%m%d%H%M%S")}

for MIN_PT in $(seq 100 100 900)
do
    MAX_PT=$(expr ${MIN_PT} + 100)
    python training.py \
        --min_pt=${MIN_PT} \
        --max_pt=${MAX_PT} \
        --max_depth=4 \
        --feature_names "ptD" "axis1" "axis2" "cmult" "nmult"
done

ARCH_DIR="./Archive/${LOG_NAME}"
mkdir -v ${ARCH_DIR}
mv -v ./logs/* ${ARCH_DIR}

python auc_heatmap.py --log_dir=${ARCH_DIR}
