#!/bin/bash -l

# Specify the log file path
log_file="fit_log.txt"

# Clear the log file before running the script
> "$log_file"

onames=("J0918+1207" "J1402+1540") #"J0918+1207" "J1402+1540")
length=${#onames[@]}
echo "$length"

for ((i=0; i<length; i++)); do
    objectName=${onames[i]}
    python3 fit.py --inDir "/home/insepien/research-data/agn-result/fit/fit_correct/masked_image_with_header/" \
    --inFile "${objectName}.fits" --oname "$objectName" --fit \
    --outDir "/home/insepien/research-data/agn-result/fit/fit_correct" \
    >> "$log_file" 2>&1
    echo "Done: $objectName"
done

