from src.crossDouble import seconds_to_text


def results_to_logfile(molecules, predictions, deviations, target_property, molecule_file, time_elapsed, save_folder):
    filename = str(save_folder + "/test_results.log")
    f = open(filename, "w")
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
