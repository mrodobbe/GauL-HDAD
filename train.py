from src.representation import gaul_representation, represent
import numpy as np
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, Input, LeakyReLU
from tensorflow.keras.initializers import glorot_uniform, RandomUniform
from tensorflow.keras.callbacks import EarlyStopping, Callback
import pickle
import sys
from src.makeMolecule import molecule_list, add_radical, add_mw, add_n, add_rings
from src.makeModel import model_builder
from src.gaussian import gmm
import time
import os
from src.plots import performance_plot, histogram_plots, gmm_plot

start = time.time()

try:
    molecule_file = sys.argv[1]
    target_property = str(sys.argv[2])
except IndexError:
    print("Not enough input files.\nPlease use the following command structure:\n"
          "python train.py <molecule_file> <property>")
    raise

try:
    molecules, outputs, bad_molecules = molecule_list(molecule_file)
except FileNotFoundError:
    print("The input file does not exist.")
    raise

# TODO: Add argument for user-defined output folder
save_folder = "/Output"
os.mkdir(save_folder)
if len(bad_molecules) > 0:
    np.savetxt("/Output/bad_molecules.txt", bad_molecules, fmt="%s")
    print("Dumped a list with molecules which could not be parsed in Output/bad_molecules.txt")
else:
    print("All molecules can be parsed by RDKit")

gmm_dict = gmm(molecules)
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

for key in histogram_dict:
    v = histogram_dict[key]
    t = gmm_dictionary[key]
    if len(key) == 2:
        metric = "Distance"
        unit = "Å"
    else:
        metric = "Angle"
        unit = "rad"
    gmm_plot(v, t, title=key, folder=str(save_folder + "/gmm"), metric=metric, unit=unit)
    histogram_plots(v, title=key, folder=str(save_folder + "/hist"), metric=metric, unit=unit)

print("Created plots and saved them!")
print("Start representing the molecules!")
representations = represent(molecules, gmm_dictionary)
print("Finished representing the molecules")

with open(str(save_folder + "/representations.pickle"), "wb") as f:
    pickle.dump(representations, f)
print("Dumped the molecule representations!")

if len(outputs.shape) == 2:
    output_layer_size = outputs.shape[1]
else:
    output_layer_size = 1

end = time.time()

print("This took {} seconds".format(round(end - start, 2)))
