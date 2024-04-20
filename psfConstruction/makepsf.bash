#!/bin/bash -l

files=$(ls ../../agn-data)

for file in $files; do
    python3 makepsf.py --inFile "$file" --makePSF
done

