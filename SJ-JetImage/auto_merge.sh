low_pt=200
for high_pt in $(seq 300 100 900)
do
    echo "${low_pt} < pT < ${high_pt} GeV"
    ./shuffle_and_merge "../Data/root_${low_pt}_${high_pt}/1-Analysed/"
    low_pt=${high_pt}
done
