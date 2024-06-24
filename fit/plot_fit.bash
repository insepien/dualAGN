#!/bin/bash -l

# Specify the log file path
log_file="plot_fit_log.txt"

# Clear the log file before running the script
> "$log_file"

#files=$(ls fit_pkls)
files=(
    'J0323+0018'
    'J0328-0710'
    'J0400-0652'
    'J0752+2019'
    'J0813+0905'
    'J0820+1801'
    'J0821+1450'
    'J0859+1001'
    'J0901+1815'
    'J0906+1840'
    'J0912+0148'
    'J0918+1100'
)
for file in "${files[@]}"; do
    # Redirect both stdout and stderr to the log file
    python3 plot_fit.py --inDir "fit_pkls/all_mod" --outDir "fit_plots/all_mod_plot" --inFile "${file}.pkl" >> "$log_file" 2>&1
    echo "Done: $file"
done
