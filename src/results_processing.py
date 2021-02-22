from src.crossDouble import seconds_to_text
from src.makeModel import model_builder
from src.plots import performance_plot
import numpy as np


def test_results_to_logfile(molecules, predictions, deviations, target_property, molecule_file, time_elapsed, save_folder):
    filename = str(save_folder + "/test_results.log")
    f = open(filename, "w")
    license_to_log(f)
    property_dict = {"h": "standard enthalpy of formation",
                     "s": "standard entropy",
                     "cp": "heat capacity"}
    if target_property in property_dict:
        f.write("GauL HDAD was able to make {} predictions.\n".format(property_dict[target_property]))
    else:
        f.write("GauL HDAD was able to make {} predictions.\n".format(target_property))
    f.write("Found input file:\t{}\n".format(molecule_file))
    f.write("Number of found molecules:\t{}\n\n".format(len(molecules)))
    f.write("================================\n")
    f.write("Molecule\tPrediction\tDeviation\n")
    for m, p, d in zip(molecules, predictions, deviations):
        f.write(str(m + "\t" + str(round(p, 2)) + "\t" + str(round(d, 2)) + "\n"))
    f.write("\nPredictions were made in {} ".format(seconds_to_text(time_elapsed)))
    f.close()


def train_results_to_logfile(molecules, outputs, results_list, representations,
                             target_property, molecule_file, time_elapsed, save_folder):
    filename = str(save_folder + "/train_results.log")
    f = open(filename, "w")
    license_to_log(f)
    property_dict = {"h": "standard enthalpy of formation",
                     "s": "standard entropy",
                     "cp": "heat capacity"}
    if target_property in property_dict:
        f.write("GauL HDAD was able to train a {} ensemble model.\n".format(property_dict[target_property]))
    else:
        f.write("GauL HDAD was able to train a {} ensemble model.\n".format(target_property))
    f.write("Found input file:\t{}\n".format(molecule_file))
    f.write("Number of training molecules:\t{}\n\n".format(len(molecules)))
    errors = np.asarray([float(line[4]) for line in results_list[1:]]).astype(np.float)
    predictions = np.asarray([float(line[2]) for line in results_list[1:]]).astype(np.float)
    real = np.asarray([float(line[1]) for line in results_list[1:]]).astype(np.float)
    performance_plot(real, predictions, "test", target_property, folder=save_folder, fold="all", model="ANN")
    mae = np.average(errors)
    rmse = np.sqrt(np.average(errors ** 2))
    f.write("Ensemble MAE: {}\n".format(round(mae, 2)))
    f.write("Ensemble RMSE: {}\n".format(round(rmse, 2)))
    f.write("================================================================================================\n")
    if len(outputs.shape) == 2:
        output_layer_size = outputs.shape[1]
    else:
        output_layer_size = 1
    model = model_builder(representations, output_layer_size, target_property)
    model.summary(print_fn=lambda x: f.write(x + '\n'))
    f.write("\nIndividual molecule predictions:\n")
    for line in results_list:
        f.write(str(str(line[0]) + "\t" + str(line[1]) + "\t" + str(line[2]) + "\t" +
                    str(line[3]) + "\t" + str(line[4]) + "\n"))
    f.write("================================================================================================\n")
    f.write("\nPredictions were made in {}.".format(seconds_to_text(time_elapsed)))
    f.close()


def license_to_log(f):
    f.write("================================================================================================\n"
            "Copyright (C) 2021  Maarten R. Dobbelaere\n\n"
            "GauL-HDAD is free software: you can redistribute it and/or modify\n"
            "it under the terms of the GNU General Public License as published by\n"
            "the Free Software Foundation, either version 3 of the License, or\n"
            "(at your option) any later version.\n\n"
            "GauL-HDAD is distributed in the hope that it will be useful,\n"
            "but WITHOUT ANY WARRANTY; without even the implied warranty of\n"
            "MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the\n"
            "GNU General Public License for more details.\n\n"
            "You should have received a copy of the GNU General Public License\n"
            "along with this program.  If not, see <https://www.gnu.org/licenses/>.\n\n"
            "When using GauL-HDAD for your own publication, please refer to the original paper:\n"
            "Learning Molecular Representations for Thermochemistry "
            "Prediction of Cyclic Hydrocarbons and Oxygenates\n"
            "Dobbelaere, M.R.; Plehiers, P.P.; Van de Vijver, R.; Stevens, C.V.; Van Geem, K.M.\n"
            "Submitted to Journal of Physical Chemistry A, 2021\n"
            "================================================================================================\n\n")
