#!/bin/bash -l

# Specify the log file path
log_file="indiv_log.txt"

# Clear the log file before running the script
> "$log_file"

objectName='J0918+1207'

python3 fit.py --inDir "/home/insepien/research-data/agn-result/fit/fit_masked_n.3to6/masked_image_SS/" \
--inFile "${objectName}.fits" --oname "$objectName" \
--outDir "/home/insepien/research-data/agn-result/fit/fit_masked_n.3to6/masked_fit" \
--PA 10 --ELL 0.5 --RE 3 --X1 53 --Y1 32   >> "$log_file" 2>&1

python3 makePlotComps.py --oname "$objectName" --sma 20 \
--inDir "/home/insepien/research-data/agn-result/fit/fit_masked_n.3to6/masked_fit" \
--outDir "/home/insepien/research-data/agn-result/fit/fit_masked_n.3to6/masked_fit_comp" >> "$log_file" 2>&1

python3 plot_fit_with1d.py --oname "$objectName" \
--inDir "/home/insepien/research-data/agn-result/fit/fit_masked_n.3to6/masked_fit_comp" \
--outDir "/home/insepien/research-data/agn-result/fit/fit_masked_n.3to6"  >> "$log_file" 2>&1

