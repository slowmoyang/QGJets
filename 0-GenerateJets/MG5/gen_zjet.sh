#!/bin/sh
low_pt=100
for high_pt in $(seq 200 100 1000)
do

    # Make output dir
    out_dir="../../Data/root_${low_pt}_${high_pt}"

    if [ -f "${out_dir}" ]
    then
        echo "${out_dir} found."
    else
        echo "${out_dir} not found."
        mkdir -v $out_dir
    fi

    # generate
    for i in $(seq 1 10)
    do
        bash generate.sh "zq" ${low_pt} ${high_pt} $i ${out_dir}
        bash generate.sh "zg" ${low_pt} ${high_pt} $i ${out_dir}
    done

    # 
    low_pt=${high_pt}

done


