#!/bin/bash -l

# Specify the log file path
log_file="fit_log.txt"

# Clear the log file before running the script
> "$log_file"

files=$(ls ../cutouts/data)

for file in $files; do
    # Redirect both stdout and stderr to the log file
    python3 fit.py --inFile "$file" >> "$log_file" 2>&1
    echo "Completed processing file: $file"
done