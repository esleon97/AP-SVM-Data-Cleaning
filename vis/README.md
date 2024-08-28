# Visualizing AP-SVM

This directory contains scripts and notebooks for visualizing the AP-SVM data cleaning model in 3D.

## Usage

### 1. Generate t-SNE Grid Data

To generate a grid of different t-SNE hyperparameter combinations, run the `tsne_grid_maker.py` script. If running in a computing cluster, you can submit this script with SLURM via:

```bash
sbatch slurm_vis.sh
```

Make sure to uncomment the appropriate line in `slurm_vis.sh`:

```sh
python tsne_grid_maker.py --file ../data/train_data_dsp.h5
```

### 2. Choose t-SNE Hyperparameters

Open the  `TSNEGridPlotter.py` script to plot the generated grid of t-SNE 3D representations. Choose the set of hyperparameters that give the best separation of clusters and save them into the `../data/hyperparameters.json` file. 


### 3. Perform 3D SVM Hyperparameter Search

To perform a random hyperparameter search for the 3D SVM, run the `3d_svm_search.py` script. If running in a computing cluster, you can submit this script with SLURM via:

```bash
sbatch slurm_vis.sh
```

Make sure to uncomment the appropriate line in `slurm_vis.sh`:

```bash
python 3d_svm_search.py --file ../data/train_data_dsp.h5
```

### 4. Create 3D SVM Mesh

To create a 3D SVM mesh, run the `3d_svm_mesh_maker.py` script. If running in a computing cluster, you can submit this script with SLURM via:

```bash
sbatch slurm_vis.sh
```

Make sure to uncomment the appropriate line in `slurm_vis.sh`:

```bash
python 3d_svm_mesh_maker.py --file ../data/train_data_dsp.h5 --npoints 300
```

### 4. Plot 3D SVM Mesh

To plot the 3D SVM mesh, run the `3d_svm_mesh_plotter.py` script. If running in a computing cluster, you can submit this script with SLURM via:

```bash
sbatch slurm_vis.sh
```
Make sure to uncomment the appropriate line in `slurm_vis.sh`:

```bash
python 3d_svm_mesh_plotter.py --file ../data/train_data_dsp.h5
```

The resulting plot of the training dataset in 3D along with the SVM decision regions will be saved in the `../plots` directory. 
