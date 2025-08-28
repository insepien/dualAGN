#!/bin/bash -l

# Specify the log file path
log_file="indiv_log.txt"

# Clear the log file before running the script
> "$log_file"

objectName='J1215+1344'

python3 fit.py --oname "$objectName" --fit \
--outDir "/home/insepien/research-data/agn-result/fit/fit_kpcbox" \
>> "$log_file" 2>&1

python3 makePlotComps.py --oname "$objectName" \
--inDir "/home/insepien/research-data/agn-result/fit/fit_kpcbox" \
--outDir "/home/insepien/research-data/agn-result/fit/fit_kpcbox" \
>> "$log_file" 2>&1

# # --paper --outFile "${objectName}.png" --modelName "sersic+sersic,sersic" \
python3 plot_fit_with1d.py --oname "$objectName" \
--outDir "/home/insepien/research-data/agn-result/fit/fit_kpcbox" \
--compDir "/home/insepien/research-data/agn-result/fit/fit_kpcbox" \
--fitDir "/home/insepien/research-data/agn-result/fit/fit_kpcbox/" \
--fitsDir "/home/insepien/research-data/agn-result/box/kpcbox/" \
 >> "$log_file" 2>&1




