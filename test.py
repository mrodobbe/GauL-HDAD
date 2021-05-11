from src.representation import load_representations
import numpy as np
import sys
from src.makeMolecule import molecule_test_list, input_checker, store_bad_molecules, denormalize, heavy_atoms
from src.results_processing import test_results_to_logfile
import time
from tensorflow.keras.models import load_model

start = time.time()

input_checker(sys.argv, "test")

molecule_file = sys.argv[1]
target_property = str(sys.argv[2])
save_folder = sys.argv[3]

molecules, conformers, bad_molecules = molecule_test_list(molecule_file)
store_bad_molecules(bad_molecules, save_folder)
representations, molecules = load_representations(molecules, conformers, save_folder)
heavy = [heavy_atoms(mol) for mol in molecules]

# TODO: detect number of folds
folders = ['Fold 1', 'Fold 2', 'Fold 3', 'Fold 4', 'Fold 5', 'Fold 6', 'Fold 7', 'Fold 8', 'Fold 9', 'Fold 10']
models = [load_model(str(save_folder + '/' + folder)) for folder in folders]

x_test = representations

ensemble = np.array([])
for model in models:
    test_predicted = model.predict(x_test).reshape(-1)
    test_predicted = np.asarray(test_predicted).astype(np.float)
    test_predicted = denormalize(test_predicted, heavy, target_property)
    if len(ensemble) == 0:
        ensemble = test_predicted
    else:
        ensemble = np.vstack((ensemble, test_predicted))

ensemble_prediction = np.mean(ensemble, axis=0)
ensemble_sd = np.std(ensemble, axis=0)



end = time.time()
time_elapsed = end-start

test_results_to_logfile(molecules, ensemble_prediction, ensemble_sd,
                        target_property, molecule_file, time_elapsed, save_folder)

# with open(str(save_folder + "/test_predictions.txt"), "w") as f:
#     f.write(str("Molecule \t Prediction \t Deviation \n"))
#     for m, p, s in zip(molecules, ensemble_prediction, ensemble_sd):
#         f.write(str(m) + '\t' + str(round(p, 4)) + '\t' + str(round(s, 4)) + '\n')
#     f.close()

