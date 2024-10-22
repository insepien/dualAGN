#!/bin/bash -l

# Specify the log file path
log_file="indiv_log.txt"

# Clear the log file before running the script
> "$log_file"

objectName='J0752+2019'

#python3 fit.py --oname "$objectName" >> "$log_file" 2>&1
python3 makePlotComps.py --oname "$objectName" --sma 10 >> "$log_file" 2>&1
python3 plot_fit_with1d.py --oname "$objectName" >> "$log_file" 2>&1
