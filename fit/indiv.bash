#!/bin/bash -l

# Specify the log file path
log_file="indiv_log.txt"

# Clear the log file before running the script
> "$log_file"

objectName='J1402+1540'
# 47,74

# python3 fit.py --inDir "./" \
# --inFile "${objectName}.fits" --oname "$objectName" --fit \
# --outDir "." \
# --PA 90 --ELL 0.4 --RE 3 --X1 47 --Y1 74 \
# >> "$log_file" 2>&1


# python3 makePlotComps.py --oname "$objectName" --sma 10 \
# --inDir "/home/insepien/research-data/agn-result/fit/fit_correct" \
# --outDir "/home/insepien/research-data/agn-result/fit/fit_correct" \
# >> "$log_file" 2>&1

# # --paper --outFile "${objectName}.png" --modelName "sersic+sersic,sersic" \
# # --outDir "/home/insepien/research-data/agn-result/fit/fit_masked_n.3to6/paper_plot" \
# --inDir "/home/insepien/research-data/agn-result/fit/fit_masked_n.3to6/masked_fit_comp" \
python3 plot_fit_with1d.py --oname "$objectName" \
--outDir "/home/insepien/research-data/agn-result/fit/fit_correct/paper_plot" \
--compDir "/home/insepien/research-data/agn-result/fit/fit_correct" \
--fitDir "/home/insepien/research-data/agn-result/fit/fit_correct/" \
--fitsDir "/home/insepien/research-data/agn-result/fit/fit_correct/masked_image_with_header/" \
--paper --outFile "${objectName}_psf.png" --modelName "exp+sersic+psf" \
 >> "$log_file" 2>&1




