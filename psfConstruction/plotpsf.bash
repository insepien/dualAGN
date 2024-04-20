#!/bin/bash -l

files=$(ls psf_pkls)

for file in $files; do
    python3 plot_psf.py --inFile "$file" 
done

