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

python chainPF.py --numsteps 500 --chainFile chain_n1_0_1000_pf.pkl --modelNum 0
