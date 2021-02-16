from src.representation import represent
import numpy as np
import sys
from src.makeMolecule import molecule_list, normalize, input_checker, store_bad_molecules
from src.crossDouble import run_cv, training, write_statistics
from src.gaussian import gmm
import time
from src.plots import output_plot
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

gmm_dictionary = gmm(molecules, conformers, save_folder)

representations, bad = represent(molecules, conformers, gmm_dictionary, save_folder)
molecules = np.delete(molecules, bad)
outputs = np.delete(outputs, bad)

outputs, heavy_atoms = normalize(molecules, outputs, target_property)

n_folds = 10  # TODO: Make argument
cv_info = training(molecules, heavy_atoms, representations, outputs, save_folder, target_property, n_folds)

end = time.time()
time_elapsed = end-start

write_statistics(cv_info, target_property, n_folds, time_elapsed, save_folder)
