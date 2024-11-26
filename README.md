# AP-SVM Data Cleaning

The Affinity Propagation (AP) + Support Vector Machine (SVM) Data Cleaning model is designed to remove anomalous and keep physical signals captured by Germanium detectors through a clustering + classification mechanism.

## Software Requirements

Create a conda environment from the `requirements.txt` file with the following command:
```bash
conda create --name apsvm --file requirements.txt
```
Make sure to run the scripts and Jupyter notebooks of this repository from the `apsvm` conda environment.

## Repository Structure

- **data/**: Contains training and testing data, configuration JSON files, and serialized model and data files produced when training the AP-SVM model.
- **plots/**: Contains plots generated during the training and testing of the AP-SVM model.
- **test/**: Contains notebooks to evaluate the performance of the AP-SVM model on test data, including sacrifice and leakage studies.
- **train/**: Contains scripts and notebooks for training and optimizing the AP-SVM model.
- **vis/**: Contains scripts and notebooks for visualizing the AP-SVM model in 3D.

## Usage

### 1. Data Preparation

Open the `data/` directory. There you will find instructions on how to acces and process the data before feeding it into AP-SVM. 

### 2. Training

Open the `train/` directory. There you will find instructions on how to train and optimize AP and SVM.  

### 4. Visualizing

Open the `vis/` directory. There you will find instructions on how to create a 3D plot of the training dataset and the SVM decision regions.

### 4. Testing

Open the `test/` directory. There you will find instructions on how to test the AP-SVM's performance and perform sacrifice and leakage studies. 


## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.

## Contact and Support

For any questions, issues, or feedback please contact [Esteban León](mailto:esleon97@unc.edu).