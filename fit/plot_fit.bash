#!/bin/bash -l

# Specify the log file path
log_file="plot_fit_log.txt"

# Clear the log file before running the script
> "$log_file"

onames=("J1222-0007" "J0932+1611") #"J0918+1207" "J1402+1540")
length=${#onames[@]}

for ((i=0; i<length; i++)); do
    objectName=${onames[i]}
    python3 makePlotComps.py --oname "$objectName" \
    --inDir "/home/insepien/research-data/agn-result/fit/fit_correct" \
    --outDir "/home/insepien/research-data/agn-result/fit/fit_correct" \
    >> "$log_file" 2>&1

    python3 plot_fit_with1d.py --oname "$objectName" \
    --outDir "/home/insepien/research-data/agn-result/fit/fit_correct" \
    --compDir "/home/insepien/research-data/agn-result/fit/fit_correct" \
    --fitDir "/home/insepien/research-data/agn-result/fit/fit_correct/" \
    --fitsDir "/home/insepien/research-data/agn-result/fit/fit_correct/masked_image_with_header/" \
     >> "$log_file" 2>&1

    echo "Done: $objectName"
done