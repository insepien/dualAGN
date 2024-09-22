#!/bin/bash -l

files=$(ls ~/agn-result/box/200 | sort)

cutSizes=(120.,  90., 176.,  75., 145.,  90., 160., 193., 109.,  90.,  99.,
       156., 109.,  82.,  68., 132., 109., 212., 160.,  90., 213., 120.,
       132., 109., 176.,  99., 120.,  90., 120.,  60., 240., 109., 193.,
       132., 160.,  90.,  90., 132., 132., 176.,  90.,  99.)


log_file="cutout_log.txt"

i=0

for f in $files; do 
    base_name=$(basename "$f" .fits)
    first_ten=${base_name:0:10}
    cutSize=${cutSizes[$i]}
    python3 make_cutout.py --outDir "~/agn-result/box/final_cut" --cutSize $cutSize --objectName "$first_ten" >> "$log_file" 2>&1
    echo "Done: $first_ten"
    i=$((i + 1))

done