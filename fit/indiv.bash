#!/bin/bash -l

# Specify the log file path
log_file="indiv_log.txt"

# Clear the log file before running the script
> "$log_file"

objectName='J1402+1540'

# python3 fit.py --oname "$objectName" --fit \
# --fitsDir "/home/insepien/research-data/agn-result/box/kpcbox" \
# --outDir "." \
# >> "$log_file" 2>&1

# python3 makePlotComps.py --oname "$objectName" \
# --inDir "/home/insepien/research-data/agn-result/fit/fit_kpcbox/kpcbox_fit" \
# --outDir "/home/insepien/research-data/agn-result/fit/fit_kpcbox/kpcbox_comp" \
# >> "$log_file" 2>&1

# # --paper --outFile "${objectName}.png" --modelName "sersic+sersic,sersic" \
# --outDir "/home/insepien/research-data/agn-result/fit/fit_kpcbox/paper" \
python3 plot_fit_with1d.py --oname "$objectName" \
--outDir "." --addModelName \
--compDir "/home/insepien/research-data/agn-result/fit/fit_kpcbox/kpcbox_comp" \
--fitDir "/home/insepien/research-data/agn-result/fit/fit_kpcbox/kpcbox_fit/" \
--fitsDir "/home/insepien/research-data/agn-result/box/kpcbox/" \
--paper --outFile "${objectName}_psf.png" --modelName "exp+sersic+psf" \
 >> "$log_file" 2>&1




