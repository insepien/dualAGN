#!/bin/bash -l

# Specify the log file path
log_file="fit_log.txt"

# Clear the log file before running the script
> "$log_file"

files=$(ls ~/agn-result/box/200 | sort)

#for f in $(echo "$files" | head -n 5); do
for f in $(echo "$files" | sed -n '31,43p'); do
    base_name=$(basename "$f" .fits)
    objectName=${base_name:0:10}
    python3 fit.py --oname "$objectName" >> "$log_file" 2>&1
    echo "Done: $objectName"
done