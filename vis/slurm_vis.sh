#!/bin/bash
#SBATCH --job-name=tsne_grid_maker  #Change according to the script you run
#SBATCH --nodes=1
#SBATCH --partition=general
#SBATCH --time=10:00:00
#SBATCH --mem=128g
#SBATCH --export=HDF5_USE_FILE_LOCKING=FALSE
#SBATCH --chdir=<path-to-vis>
#SBATCH --output=<path-to-vis>/tsne_grid_maker_%j.txt #Change according to the script you run
#SBATCH --mail-type=begin,end,fail
#SBATCH --mail-user=<your-email>

echo "Job Start:"
date
echo "Node(s):  "$SLURM_JOB_NODELIST
echo "Job ID:  "$SLURM_JOB_ID

# Activate the conda environment
conda activate apsvm
# Uncomment to run the desired script
python tsne_grid_maker.py --file ../data/train_data_dsp.h5
# python 3d_svm_search.py --file ../data/train_data_dsp.h5
# python 3d_svm_mesh_maker.py --file ../data/train_data_dsp.h5 --npoints 300
# python 3d_svm_mesh_plotter_detector.py --file ../data/train_data_dsp.h5



echo "Job Complete:"
date
