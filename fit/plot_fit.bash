#!/bin/bash -l

# Specify the log file path
log_file="plot_fit_log.txt"

# Clear the log file before running the script
> "$log_file"

mapfile -t files < <(ls ~/research-data/agn-result/fit/fit_masked_n.3to6/masked_fit | sort)
#files=("${files[@]:24}")
length=39

#for f in $(echo "$files" | head -n 5); do
for ((i=0; i<length; i++)); do
    f=${files[i]}
    base_name=$(basename "$f" .fits)
    objectName=${base_name:0:10}
    #python3 makePlotComps.py --oname "$objectName" --sma 10 --inDir "~/research-data/agn-result/fit/test_fit_masked/masked_fit" --outDir "~/research-data/agn-result/fit/test_fit_masked/masked_fit_comp" >> "$log_file" 2>&1
    python3 plot_fit_with1d.py --oname "$objectName" \
    --inDir "/home/insepien/research-data/agn-result/fit/fit_masked_n.3to6/masked_fit_comp" \
    --outDir "/home/insepien/research-data/agn-result/fit/fit_masked_n.3to6" \
    >> "$log_file" 2>&1
    echo "Done: $objectName"
done