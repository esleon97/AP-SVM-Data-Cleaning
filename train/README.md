# Training AP-SVM

This directory contains scripts and notebooks for training and optimizing the AP-SVM data cleaning model.

## Usage

### 1. AP Hyperparameter Optimization 

Find the optimal hyperparameters for AP by running the ``ap_search.py`` script. If running in a computing cluster, you can submit this optimization script with SLURM via:

```bash
sbatch slurm_ap.sh
```

### 2. AP Re-Labeling

Open the `APBrowser.ipynb` notebook to re-label the clusters found by AP in terms of data cleaning categories.

### 3. SVM Hyperparameter Optimization

Find the optimal hyperparameters for SVM by running the ``svm_search.py`` script. If running in a computing cluster, you can submit this optimization script with SLURM via:

```bash
sbatch slurm_svm.sh
```

### 4. SVM Final Training

Open the `SVMBrowser.ipynb` notebook to train the SVM with optimal hyperparameters and save it as a `.sav` file in the `../data` directory.