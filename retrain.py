from src.representation import represent
import numpy as np
import pickle
import sys
from src.makeMolecule import molecule_list, normalize, heavy_atoms
from src.crossValidation import run_cv
import time
from sklearn.model_selection import KFold
from joblib import Parallel, delayed, cpu_count

start = time.time()

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
        representations = represent(molecules, gmm_dictionary)
    except FileNotFoundError:
        print("No gmm dictionary found. Select another folder or use train.py")
        raise

if target_property != "h":
    outputs, heavy_atoms = normalize(molecules, outputs, "s", coefficient=1.5)
else:
    heavy_atoms = np.asarray([heavy_atoms(mol) for mol in molecules])

n_folds = 10  # TODO: Make argument
kf = KFold(n_folds, shuffle=True, random_state=12081997)

# cpu = cpu_count()
cpu = 2
if n_folds > cpu:
    n_jobs = cpu
else:
    n_jobs = n_folds

prediction_mae = []
prediction_rmse = []
prediction_svr_mae = []
prediction_svr_rmse = []

test_models = []
test_svr = []

cv_info = Parallel(n_jobs=n_jobs)(delayed(run_cv)(molecules, heavy_atoms, representations, outputs,
                                                  loop_kf, i, save_folder, target_property)
                                  for loop_kf, i in zip(kf.split(representations), range(1, n_folds+1)))

for j in range(n_folds):
    prediction_mae.append(cv_info[j][0])
    prediction_rmse.append(cv_info[j][1])
    if target_property != "cp":
        prediction_svr_mae.append(cv_info[j][2])
        prediction_svr_rmse.append(cv_info[j][3])

best_index = np.argmin(prediction_rmse)
if target_property != "cp":
    best_index_svr = np.argmin(prediction_svr_rmse)
    with open(str(save_folder + "/best_models.txt"), "w") as f:
        f.write("The best ANN model is from fold {}, "
                "while the best SVR model is from fold {}\n".format(best_index + 1, best_index_svr + 1))
        f.close()
else:
    with open(str(save_folder + "/best_models.txt"), "w") as f:
        f.write("The best ANN model is from fold {}.".format(best_index + 1))
        f.close()

end = time.time()
time_elapsed = end-start

with open(str(save_folder + "/test_statistics.txt"), "w") as f:
    f.write('Test performance statistics for ANN:\n')
    f.write('Mean absolute error:\t\t{:.2f} +/- {} kJ/mol\n'.format(np.mean(prediction_mae), np.std(prediction_mae)))
    f.write('Root mean squared error:\t{:.2f} +/- {} kJ/mol\n\n'.format(np.mean(prediction_rmse),
                                                                        np.std(prediction_rmse)))
    for i, mae_value, rmse_value in zip(range(len(prediction_mae)), prediction_mae, prediction_rmse):
        f.write('Fold {} - MAE: {} kJ/mol\t\t-\t\tRMSE: {} kJ/mol\n'.format(i+1, mae_value, rmse_value))
    if target_property != "cp":
        f.write('\nTest performance statistics for SVR:\n')
        f.write('Mean absolute error:\t\t{:.2f} +/- {} kJ/mol\n'.format(np.mean(prediction_svr_mae),
                                                                        np.std(prediction_svr_mae)))
        f.write('Root mean squared error:\t{:.2f} +/- {} kJ/mol\n\n'.format(np.mean(prediction_svr_rmse),
                                                                            np.std(prediction_svr_rmse)))
        for i, mae_value, rmse_value in zip(range(len(prediction_svr_mae)), prediction_svr_mae, prediction_svr_rmse):
            f.write('Fold {} - MAE: {} kJ/mol\t\t-\t\tRMSE: {} kJ/mol\n'.format(i + 1, mae_value, rmse_value))
    f.write('\nTime elapsed: {} seconds\n'.format(time_elapsed))
    f.close()

print("Finished! This took {} seconds".format(round(time_elapsed, 1)))