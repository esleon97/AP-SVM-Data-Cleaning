#!/bin/bash
#SBATCH --job-name=ap_search
#SBATCH --nodes=1
#SBATCH --partition=general
#SBATCH --time=04:00:00
#SBATCH --ntasks=20  #This number should be equal to (or a factor of) the number of grid points
#SBATCH --mem-per-cpu=16g
#SBATCH --export=HDF5_USE_FILE_LOCKING=FALSE
#SBATCH --chdir=<path-to-train>
#SBATCH --output=<path-to-train>/ap_search_%j.txt
#SBATCH --mail-type=begin,end,fail
#SBATCH --mail-user=<your-email>

echo "Job Start:"
date
echo "Node(s):  "$SLURM_JOB_NODELIST
echo "Job ID:  "$SLURM_JOB_ID

# Activate the conda environment
conda activate apsvm
# Run the AP optimization
python ap_search.py --file ../data/train_data_dsp.h5 --exemplars 100 --prefs 10 --damps 10


echo "Job Complete:"
date
