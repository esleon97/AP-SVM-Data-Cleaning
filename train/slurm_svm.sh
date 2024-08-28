#!/bin/bash
#SBATCH --job-name=svm_search
#SBATCH --nodes=1
#SBATCH --partition=general
#SBATCH --time=04:00:00
#SBATCH --mem=64g
#SBATCH --export=HDF5_USE_FILE_LOCKING=FALSE 
#SBATCH --chdir=<path-to-train>
#SBATCH --output=<path-to-train>/svm_search_%j.txt
#SBATCH --mail-type=begin,end,fail
#SBATCH --mail-user=<your-email>

echo "Job Start:"
date
echo "Node(s):  "$SLURM_JOB_NODELIST
echo "Job ID:  "$SLURM_JOB_ID

# Activate the conda environment
conda activate apsvm
# Run the SVM optimization
python svm_search.py --file ../data/train_data_dsp.h5

echo "Job Complete:"
date
