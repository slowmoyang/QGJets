#!/bin/sh

low_pt=100
for high_pt in $(seq 200 100 1000)
do

    # Make output dir
    out_dir="/home/slowmoyang/CMSSW_8_0_26/src/QGJets/Data/root_${low_pt}_${high_pt}"

    ls -l "/home/slowmoyang/CMSSW_8_0_26/src/QGJets/"

    if [ -f "${out_dir}" ]
    then
        echo "${out_dir} found."
    else
        echo "${out_dir} not found."
        mkdir -v ${out_dir}
        chmod 757 ${out_dir}
    fi

    # generate
    for i in $(seq 1 3)
    do
        bash generate.sh "qq" ${low_pt} ${high_pt} $i ${out_dir}
        bash generate.sh "gg" ${low_pt} ${high_pt} $i ${out_dir}
    done

    # 
    low_pt=${high_pt}

done
