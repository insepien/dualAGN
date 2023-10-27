#!/bin/bash -l

#SBATCH --account=kmpardo_1142
#SBATCH --partition=main
#SBATCH --time 10:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=16G
#SBATCH --mail-type=END
#SBATCH --mail-user=hamydo@usc.edu

conda activate agn

python indiv-chainPF.py --numsteps 1000 --chainFile chain_n1m0.pkl --fitFile J1215+1344_n1m0_fit.pkl
 
