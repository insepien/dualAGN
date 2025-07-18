#!/bin/bash -l

# Specify the log file path
log_file="indiv_log.txt"

# Clear the log file before running the script
> "$log_file"

objectName='J1402+1540'

# python3 fit.py --inDir "/home/insepien/research-data/agn-result/fit/fit_correct/masked_image_with_header/" \
# --inFile "${objectName}.fits" --oname "$objectName" --fit \
# --outDir "/home/insepien/research-data/agn-result/fit/fit_correct" \
# --PA 45 --ELL 0.2 --RE 3 \
# >> "$log_file" 2>&1


# python3 makePlotComps.py --oname "$objectName" \
# --inDir "/home/insepien/research-data/agn-result/fit/fit_correct" \
# --outDir "/home/insepien/research-data/agn-result/fit/fit_correct" \
# >> "$log_file" 2>&1

# # --paper --outFile "${objectName}.png" --modelName "sersic+sersic,sersic" \
# # --outDir "/home/insepien/research-data/agn-result/fit/fit_masked_n.3to6/paper_plot" \
# --inDir "/home/insepien/research-data/agn-result/fit/fit_masked_n.3to6/masked_fit_comp" \
python3 plot_fit_with1d.py --oname "$objectName" \
--outDir "/home/insepien/research-data/agn-result/fit/fit_correct" \
--compDir "/home/insepien/research-data/agn-result/fit/fit_correct" \
--fitDir "/home/insepien/research-data/agn-result/fit/fit_correct/" \
--fitsDir "/home/insepien/research-data/agn-result/fit/fit_correct/masked_image_with_header/" \
--paper --outFile "${objectName}_ser.png" --modelName "sersic+sersic,sersic+sersic" \
 >> "$log_file" 2>&1




