#!/bin/bash -l

# Specify the log file path
log_file="indiv_log.txt"

# Clear the log file before running the script
> "$log_file"

objectName='J0328-0710'

python3 fit.py --oname "$objectName" --outDir "~/research-data/agn-result/fit" --PA 10 >> "$log_file" 2>&1
python3 makePlotComps.py --oname "$objectName" --sma 20 --inDir "~/research-data/agn-result/fit" --outDir "~/research-data/agn-result/fit" >> "$log_file" 2>&1
python3 plot_fit_with1d.py --oname "$objectName" --inDir "~/research-data/agn-result/fit" --outDir "~/research-data/agn-result/fit"  >> "$log_file" 2>&1
