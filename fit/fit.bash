#!/bin/bash -l

# Specify the log file path
log_file="fit_log.txt"

# Clear the log file before running the script
> "$log_file"

mapfile -t files < <(ls ~/research-data/agn-result/box/200 | sort)
files=("${files[@]:18}")
PAs=(70 90 100 170 160 10 170)
length=${#PAs[@]}

for ((i=0; i<length; i++)); do
    f=${files[i]}
    pa=${PAs[i]} 
    base_name=$(basename "$f" .fits)
    objectName=${base_name:0:10}
    #python3 fit.py --oname "$objectName" --PA "$pa" --outDir "~/research-data/agn-result/fit/test_fit_masked/masked_fit" >> "$log_file" 2>&1
    echo "Done: $objectName"
done