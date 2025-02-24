# Framework for Test Dataset Quality Assessment in Deep Neural Networks using LSCD, DSC and MS

## Overview
This framework is used to calculate Latent Space Class Dispersion, Mutation Scores and distance-based Surprise Coverage for test dataset quality evaluation. Using this framework, mutant models using pre-training MOs can be trained. Later, LSCD, MS and  distance-based Surprise Coverage scores are calculated and their correlation is studied thereby.

### Installation
- Required python version: 3.10
- OS: Linux, Windows

### Create virtual environment
python3.10 -m venv venv

### Activate virtual environment
source venv/bin/activate

### Install basic requirements
python -m pip install -r requirements.txt


## Folder Structure

### Data

This folder contains pre-processed datasets that can be used to train mutant models in (/mutagen/).

### Dataset
This folder contains dataloading scripts that use base_dataset.py as a base class for loading the MNIST, SVHN, and GTSRB datasets. This can be extended to use a new dataset. The "corner_case_dataset.py" file contains a data loader for corner case images from fuzzing.

### Models

This folder contains original trained DNN models and their PyTorch implementations. 

### mutagen

Use this folder and associated files to create mutant models and calculate LSCD, MS, and their correlation using the Pearson Correlation coefficient. A detailed guide can be found in this folder.

### Results

This folder should contain results obtained from mutation generation.

