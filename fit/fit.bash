#!/bin/bash -l

# Specify the log file path
log_file="fit_log.txt"

# Clear the log file before running the script
> "$log_file"

mapfile -t allfiles < <(ls ~/research-data/agn-result/box/kpcbox/finalbox | sort)

files=("${allfiles[@]:0:2}")

for f in "${files[@]}"; do
    oname=${f:0:10}
    echo "$oname"
    python3 fit.py --fit --fitsDir "/home/insepien/research-data/agn-result/box/kpcbox" --oname "$oname" --outDir "/home/insepien/research-data/agn-result/fit/fit_kpcbox" >> "$log_file" 2>&1
done
