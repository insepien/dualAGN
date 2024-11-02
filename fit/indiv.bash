#!/bin/bash -l

# Specify the log file path
log_file="indiv_log.txt"

# Clear the log file before running the script
> "$log_file"

objectName='J1402+1540'

#python3 fit.py --oname "$objectName" --outDir "~/research-data/agn-result/fit/test_fit_masked" --PA 10 >> "$log_file" 2>&1
python3 makePlotComps.py --oname "$objectName" --sma 30 --inDir "~/research-data/agn-result/fit/test_fit_masked/masked_fit" --outDir "~/research-data/agn-result/fit/test_fit_masked/masked_fit_comp" >> "$log_file" 2>&1
python3 plot_fit_with1d.py --oname "$objectName" --inDir "~/research-data/agn-result/fit/test_fit_masked/masked_fit_comp" --outDir "~/research-data/agn-result/fit/test_fit_masked"  >> "$log_file" 2>&1
