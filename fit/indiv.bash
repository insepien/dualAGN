#!/bin/bash -l

# Specify the log file path
log_file="indiv_log.txt"

# Clear the log file before running the script
> "$log_file"

objectName='J1215+1344'

# python3 fit.py --inDir "/home/insepien/research-data/agn-result/fit/fit_masked_n.3to6/masked_image_SS/" \
# --inFile "${objectName}.fits" --oname "$objectName" \
# --outDir "/home/insepien/research-data/agn-result/fit/fit_masked_n.3to6/masked_fit" \
# --PA 10 --ELL 0.1 --RE 3 \
# >> "$log_file" 2>&1

# python3 makePlotComps.py --oname "$objectName" --sma 10 \
# --inDir "/home/insepien/research-data/agn-result/fit/fit_masked_n.3to6/masked_fit" \
# --outDir "/home/insepien/research-data/agn-result/fit/fit_masked_n.3to6/masked_fit_comp" \
# >> "$log_file" 2>&1

# --paper --outFile "${objectName}.png" --modelName "sersic+sersic,sersic" \
# --outDir "/home/insepien/research-data/agn-result/fit/fit_masked_n.3to6/paper_plot" \
python3 plot_fit_with1d.py --oname "$objectName" \
--inDir "/home/insepien/research-data/agn-result/fit/fit_masked_n.3to6/masked_fit_comp" \
--paper --outFile "${objectName}.png" --modelName "sersic+psf,sersic+psf" \
--outDir "/home/insepien/research-data/agn-result/fit/fit_masked_n.3to6/paper_plot" \
 >> "$log_file" 2>&1



