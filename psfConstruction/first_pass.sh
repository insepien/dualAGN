#!/bin/bash

# Store the list of files in the ~/agn directory in a variable
files=$(ls ~/agn-data)

# Loop through each file in the list
for file in $files; do
    python3 makepsf.py --inFile "$file" --firstPass
done

