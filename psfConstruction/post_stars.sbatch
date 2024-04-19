#!/bin/bash -l

#SBATCH --account=kmpardo_1142
#SBATCH --partition=main
#SBATCH --time 2:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=16G
#SBATCH --mail-type=END
#SBATCH --mail-user=hamydo@usc.edu

conda activate agn

files=$(ls ../../agn-data)

for file in $files; do
    python3 makepsf.py --inFile "$file" 
done

