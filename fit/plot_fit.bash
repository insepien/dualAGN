#!/bin/bash -l

# Specify the log file path
log_file="plot_fit_log.txt"

# Clear the log file before running the script
> "$log_file"

files=$(ls fit_pkls)
file_number=1

for file in $files; do
    # Redirect both stdout and stderr to the log file
    python3 plot_fit.py --inFile "$file" >> "$log_file" 2>&1
    echo "Completed processing file number $file_number /42"
    ((file_number++))
done