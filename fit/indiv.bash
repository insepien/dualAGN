#!/bin/bash -l

# Specify the log file path
log_file="indiv_log.txt"

# Clear the log file before running the script
> "$log_file"

objectName='J1204+0335'

# python3 fit.py --oname "$objectName" --outDir "/home/insepien/research-data/agn-result/fit/fit_masked_n0to10/masked_fit" --PA 10 >> "$log_file" 2>&1
# python3 makePlotComps.py --oname "$objectName" --sma 10 --inDir "/home/insepien/research-data/agn-result/fit/fit_masked_n0to10/masked_fit" --outDir "/home/insepien/research-data/agn-result/fit/fit_masked_n0to10/masked_fit_comp" >> "$log_file" 2>&1
# python3 plot_fit_with1d.py --oname "$objectName" --inDir "/home/insepien/research-data/agn-result/fit/fit_masked_n0to10/masked_fit_comp" --outDir "/home/insepien/research-data/agn-result/fit/fit_masked_n0to10"  >> "$log_file" 2>&1

python3 fit.py --oname "$objectName" --inDir "~/research-data/agn-result/box/final_cut/" --inFile "J1204+0335_200.fits" --outDir "/home/insepien/research-data/agn-result/fit" --PA 10 >> "$log_file" 2>&1
python3 makePlotComps.py --oname "$objectName" --sma 10 --inDir "/home/insepien/research-data/agn-result/fit" --outDir "/home/insepien/research-data/agn-result/fit" >> "$log_file" 2>&1
python3 plot_fit_with1d.py --oname "$objectName" --inDir "/home/insepien/research-data/agn-result/fit" --outDir "/home/insepien/research-data/agn-result/fit"  >> "$log_file" 2>&1
