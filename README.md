# GauL-HDAD
Machine learning method for predicting thermochemical (and many more) properties

This repository contains the GauL HDAD algorithm for representing molecules as a numeric vector and neural networks for molecular property prediction.

The full paper can be found [here](doi.org/10.1021/acs.jpca.1c01956).
If you have any questions, comments, ideas, improvements, don't hesitate to contact me! Please find contact details [here](lct.ugent.be/people/maarten-dobbelaere).


## Requirements
For small datasets (<1000 molecules), it is possible to train on a standard laptop in less than 30 minutes. It is recommended to use high-performance computing for larger datasets. Please keep in mind that training large datasets (>40000 molecules) will take more than 2 days.

GauL-HDAD is written in python and can be run from command prompt. This requires the installation of several python packages, which are listed below. It is recommended to use Anaconda for managing packages (https://docs.anaconda.com/anaconda/install/).
* Python 3.6 or higher
* NumPy         `conda install numpy`
* RDKit         `conda install -c conda-forge rdkit`
* Scikit-learn  `conda install -c conda-forge scikit-learn`
* joblib        `conda install -c anaconda joblib`
* TensorFlow 2  `pip install tensorflow`
* matplotlib    `conda install -c conda-forge matplotlib`

## Installation
GauL-HDAD can be used by cloning this repository and moving in the Anaconda terminal to the folder in which it was cloned.
* Open Anaconda Prompt (Anaconda3) from the Windows start menu
* Use the `cd` command in Anaconda Prompt to go to the folder where GauL-HDAD was cloned

## Data
Models can only be trained when data is provided. GauL-HDAD can handle `.txt` files containing molecules and target values, separated by a tab.
Three molecular formats can be parsed: SMILES, InChI and 3D coordinates in `.mol` files. When your data contains mesomeric radicals, use of InChIs is discouraged.

```
CC	100
InChI=1S/C2H6/c1-2/h1-2H3	100
ethane.mol	100

```

Although all formats can be used interchangeably, it is strongly recommended to stick to only one input type.

## Training
A model ensemble can be trained by running following command in Anaconda Prompt (in the repository folder):
```
python train.py <input_file> <target_property> <save_folder>
```
* `<input_file>` is the file location of the training data
* `<target_property>` is the property that will be modeled. 
  * Use `h` for enthalpy
  * Use `s` for entropy
  * Use `cp` for heat capacity
  * Something else can be used for another property, the models will be trained, but are possibly not optimal.
* `<save_folder>` is the folder where you want to store your results. This folder will be created as a subfolder of the repository

Example: Your enthalpy data is stored in a subfolder `Data` as `input.txt` and you want to store your results in a folder named `Results`:
```
python train.py Data/input.txt h Results
```

**WARNING:** If `save_folder` already exists, the existing data will be overwritten!

A large number of files will be created in `<save_folder>`. 
* `train_results.log` gives all information you need: model architecture, preediction statistics and individual predictions
* `output_plot.png` shows the input data as a function of the number of heavy atoms. This gives an idea about the distribution of your data.
* `hist` is a folder that contains all individual histograms.
* `gmm` is a folder that contains the Gaussian mixture models of the histograms in `hist`.
* `test_results_fold_x_y.txt` are the results for the individual folds
* `gmm_dictionary.pickle` contains the values of the Gaussian mixture models for use in new models
* `histogram_dictionary.pickle` contains the values of the histograms
* `ll_dictionary.pickle` contains the values of all log-likelihoods from Gaussian mixture modeling
* `test_statistics.txt` contains the performance measures on the test set in each ensemble fold.
* `Fold x` (x = 1 ... #folds) are folders containing data per fold:
  * a trained neural network
  * `test_results_fold_x.txt` contains the test set performance in this fold
  * `test_ensemble_predictions_x.txt` contains the predictions of the test set molecules with the "Real value" (from `input.txt`), the ensemble predicted value, the ensemble standard deviation on the prediction and the absolute error.
  
**INFO:** In the current version, nested cross-validation is implemented with 10 outer folds and 9 inner folds.

## Training with previous representations
Creating Gaussian mixture models is a very time-consuming step in the algorithm. It is therefore likely that you want to work with previous data. This is possible with using a similar command as for training:
```
python retrain.py <input_file> <target_property> <save_folder>
```
The code will search for either a `representations.pickle` file or (when that is not included) for a `gmm_dictionary.pickle` file. It is, hence, **necessary** to create a `<save_folder>` that includes one of those files from a previous run! It will raise an error if neither of those files is included.

## Making predictions
Making predictions with GauL-Thermo is easy! The command is similar to those for training and retrining:
```
python test.py <input_file> <target_property> <save_folder>
```
`<input_file>` is also a `.txt` file, but it has only one column: the molecule. All molecules must be stored in a `.txt` file all below each other.
`<save_folder>` is the folder where the model was trained that you want to use for testing.

## How to refer to this model?
When using GauL-HDAD for your own publication, please cite the original paper:

*Learning Molecular Representations for Thermochemistry Prediction of Cyclic Hydrocarbons and Oxygenates <br>
Dobbelaere, M.R.; Plehiers, P.P.; Van de Vijver, R.;  Stevens, C.V.; Van Geem, K.M. <br>
Journal of Physical Chemistry A, 2021, 125, 23, pp. 5166-5179*
