#!/bin/bash -l

# Specify the log file path
log_file="fit_log.txt"

# Clear the log file before running the script
> "$log_file"

# files=$(ls ../cutouts/data)

files=(
    'J0328-0710.fits'
    'J0400-0652.fits'
    'J0752+2019.fits'
    'J0813+0905.fits'
    'J0820+1801.fits'
    'J0821+1450.fits'
    'J0859+1001.fits'
    'J0901+1815.fits'
    'J0906+1840.fits'
    'J0912+0148.fits'
    'J0918+1100.fits'
)

for file in "${files[@]}"; do
    # Redirect both stdout and stderr to the log file
    python3 fit_allmod.py --inFile "$file" >> "$log_file" 2>&1
    echo "Done: $file"
done