from src.representation import represent
import numpy as np
import pickle
import sys
from src.makeMolecule import molecule_list
import time
from sklearn.model_selection import KFold
from joblib import Parallel, delayed, cpu_count, wrap_non_picklable_objects
from tensorflow.keras.callbacks import EarlyStopping, Callback
from tensorflow.keras import Model
from tensorflow.keras.models import load_model
from src.makeMolecule import normalize, denormalize, heavy_atoms
from sklearn.svm import SVR
from src.plots import performance_plot

try:
    molecule_file = sys.argv[1]
    target_property = str(sys.argv[2])
except IndexError:
    print("Not enough input files.\nPlease use the following command structure:\n"
          "python train.py <molecule_file> <property> <save_folder>")
    raise

try:
    save_folder = str(sys.argv[3])
except IndexError:
    print("Not enough input files.\nPlease use the following command structure:\n"
          "python train.py <molecule_file> <property> <save_folder>")
    raise

try:
    molecules, outputs, bad_molecules = molecule_list(molecule_file)
except FileNotFoundError:
    print("The input file does not exist.")
    raise
try:
    with open(str(save_folder + "/representations.pickle"), "rb") as f:
        representations = pickle.load(f)
    print("Loaded the molecule representations!")
except FileNotFoundError:
    print("No representations available! Trying to find a gmm dictionary")
    try:
        with open(str(save_folder + "/gmm_dictionary.pickle"), "rb") as f:
            gmm_dictionary = pickle.load(f)
        print("Loaded the GMM data!")
        representations, bad = represent(molecules, gmm_dictionary)
        molecules = np.delete(molecules, bad)
        outputs = np.delete(outputs, bad)
    except FileNotFoundError:
        print("No gmm dictionary found. Please include a gmm dictionary in {}".format(save_folder))
        raise

folders = ['Fold 1', 'Fold 10', 'Fold 2', 'Fold 3', 'Fold 4', 'Fold 5', 'Fold 6', 'Fold 7', 'Fold 8', 'Fold 9']
models = [load_model(str(save_folder + '/' + folder)) for folder in folders]

x_test = representations
y_test = outputs

ensemble = np.array([])
for model in models:
    test_predicted = model.predict(x_test).reshape(-1)
    test_predicted = np.asarray(test_predicted).astype(np.float)
    if len(ensemble) == 0:
        ensemble = test_predicted
    else:
        ensemble = np.vstack((ensemble, test_predicted))

ensemble_prediction = np.mean(ensemble, axis=0)
ensemble_error = np.abs(ensemble_prediction - y_test)
columns = len(ensemble[0, :])
individual_mae = []
individual_rmse = []
for i in range(columns):
    err = np.abs(y_test - ensemble[:, i])
    individual_mae.append(np.average(err))
    individual_rmse.append(np.average(err ** 2))

ensemble_mae = np.average(ensemble_error)
ensemble_rmse = np.sqrt(np.average(ensemble_error ** 2))

save_name = "/test_results_{}.txt".format(len(y_test))

with open(str(save_folder + save_name), "w") as f:
    f.write('ANN Ensemble Test performance statistics:\n')
    f.write('Mean absolute error:\t\t{:.2f} kJ/mol\n'.format(ensemble_mae))
    f.write('Root mean squared error:\t{:.2f} kJ/mol\n\n'.format(ensemble_rmse))
    for i, mae_value, rmse_value in zip(range(len(individual_mae)), individual_mae, individual_rmse):
        f.write('Fold {} - MAE: {} kJ/mol\t\t-\t\tRMSE: {} kJ/mol\n'.format(i + 1, mae_value, rmse_value))
    f.close()

