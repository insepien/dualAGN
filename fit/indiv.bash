#!/bin/bash -l

# Specify the log file path
log_file="indiv_log.txt"

# Clear the log file before running the script
> "$log_file"

objectName='J1245+0535'

#python3 fit.py --oname "$objectName" --mask >> "$log_file" 2>&1
python3 makePlotComps.py --oname "$objectName" --sma 35 >> "$log_file" 2>&1
python3 plot_fit_with1d.py --oname "$objectName" >> "$log_file" 2>&1
#python3 makePlotComps.py --oname "$objectName" --sma 20 --inDir "~/agn-result/fit/final_fit_nb" --outDir "~/agn-result/fit/final_fit_nb" >> "$log_file" 2>&1
#python3 plot_fit_with1d.py --oname "$objectName" --inDir "~/agn-result/fit/final_fit_nb" --outDir "~/agn-result/fit/final_fit_nb"  >> "$log_file" 2>&1
