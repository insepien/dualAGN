#!/bin/bash -l

# Specify the log file path
log_file="indiv_log.txt"

# Clear the log file before running the script
> "$log_file"

objectName='J1402+1540'

python3 fit.py --inDir "/home/insepien/research-data/agn-result/fit/fit_masked_n.3to6(wrongBG)/masked_image_SS/" \
--inFile "${objectName}.fits" --oname "$objectName" \
--outDir "/home/insepien/research-data/agn-result/fit/fit_correct" --fit \
--PA 10 --ELL 0.1 --RE 3 \
>> "$log_file" 2>&1


# --inDir "/home/insepien/research-data/agn-result/fit/fit_masked_n.3to6/refit_boxsize" \
# --outDir "/home/insepien/research-data/agn-result/fit/fit_masked_n.3to6/refit_boxsize" \
# python3 makePlotComps.py --oname "$objectName" --sma 20 \
# --inDir "." --outDir "." \
# >> "$log_file" 2>&1

# # --paper --outFile "${objectName}.png" --modelName "sersic+sersic,sersic" \
# # --outDir "/home/insepien/research-data/agn-result/fit/fit_masked_n.3to6/paper_plot" \
# --inDir "/home/insepien/research-data/agn-result/fit/fit_masked_n.3to6/masked_fit_comp" \
# python3 plot_fit_with1d.py --oname "$objectName" \
# --compDir "." --fitDir "./" --outDir "." --fitsDir "/home/insepien/research-data/agn-result/fit/fit_masked_n.3to6/masked_image_with_header/" \
# >> "$log_file" 2>&1
# --compDir "/home/insepien/research-data/agn-result/fit/fit_masked_n.3to6/refit_boxsize" \
# --fitDir "/home/insepien/research-data/agn-result/fit/fit_masked_n.3to6/refit_boxsize/" \
# --fitsDir "/home/insepien/research-data/agn-result/fit/fit_masked_n.3to6/refit_boxsize/" \
# --outDir "/home/insepien/research-data/agn-result/fit/fit_masked_n.3to6/refit_boxsize" \
#  >> "$log_file" 2>&1



