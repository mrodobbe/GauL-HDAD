from src.representation import represent
import numpy as np
import pickle
import sys
from src.makeMolecule import molecule_list, normalize, heavy_atoms, input_checker, store_bad_molecules
from src.crossValidation import run_cv
from src.gaussian import gmm
import time
from src.plots import histogram_plots, gmm_plot, output_plot
from sklearn.model_selection import KFold
from joblib import Parallel, delayed, cpu_count

start = time.time()

input_checker(sys.argv)

molecule_file = sys.argv[1]
target_property = str(sys.argv[2])
save_folder = sys.argv[3]

molecules, outputs, conformers, bad_molecules = molecule_list(molecule_file)
store_bad_molecules(bad_molecules, save_folder)

output_plot(molecules, outputs, target_property, folder=save_folder)

gmm_dict = gmm(molecules, save_folder)
print("Successfully finished clustering all geometry features!")
# TODO: Bad molecules must be popped from lists
# TODO: Put these lines somewhere in the background

with open(str(save_folder + "/gmm_dictionary.pickle"), "wb") as f:
    pickle.dump(gmm_dict[0], f)
print("Dumped the GMM data!")
with open(str(save_folder + "/ll_dictionary.pickle"), "wb") as f:
    pickle.dump(gmm_dict[1], f)
print("Dumped the log-likelihood data!")
with open(str(save_folder + "/histogram_dictionary.pickle"), "wb") as f:
    pickle.dump(gmm_dict[2], f)
print("Dumped the histograms!")

histogram_dict = gmm_dict[2]
gmm_dictionary = gmm_dict[0]

# TODO: Make function for plotting

# for key in histogram_dict:
#     v = histogram_dict[key]
#     t = gmm_dictionary[key]
#     if len(key) == 2:
#         metric = "Distance"
#         unit = "Å"
#     else:
#         metric = "Angle"
#         unit = "rad"
#     gmm_plot(v, t, title=key, folder=str(save_folder + "/gmm"), metric=metric, unit=unit)
#     histogram_plots(v, title=key, folder=str(save_folder + "/hist"), metric=metric, unit=unit)

# TODO: Pop bad molecules
print("Created plots and saved them!")
print("Start representing the molecules!")
representations, bad = represent(molecules, gmm_dictionary)
molecules = np.delete(molecules, bad)
outputs = np.delete(outputs, bad)
print("Finished representing the molecules")

with open(str(save_folder + "/representations.pickle"), "wb") as f:
    pickle.dump(representations, f)
print("Dumped the molecule representations!")

if target_property != "h":
    outputs, heavy_atoms = normalize(molecules, outputs, "s", coefficient=1.5)
else:
    heavy_atoms = np.asarray([heavy_atoms(mol) for mol in molecules])

n_folds = 10  # TODO: Make argument
kf = KFold(n_folds, shuffle=True, random_state=12081997)

cpu = cpu_count()
# cpu = 4
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
