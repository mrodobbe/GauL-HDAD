from src.makeMolecule import nasa_fit, molecule_test_list
import numpy as np

folder = "C:/Users/mrodobbe/Documents/Research/GauL Histograms/GauL Thermo/Results"

h_qm_file = "../Data/h_lignin_smiles.txt"
h_pred_file = "../Data/h_lignin_smiles_predictions.txt"
cp_qm_file = "../Data/cp_lignin_smiles.txt"
cp_pred_file = "../Data/cp_lignin_smiles_predictions.txt"

mol_h_qm, h_qm = molecule_test_list(h_qm_file)
mol_h_pred, h_pred = molecule_test_list(h_pred_file)
mol_cp_qm, cp_qm = molecule_test_list(cp_qm_file)
mol_cp_pred, cp_pred = molecule_test_list(cp_pred_file)

print(cp_qm)

h_dict = {}
cp_dict = {}

for m, o in zip(mol_h_qm, h_qm):
    h_dict[m] = [o]

for m, o in zip(mol_h_pred, h_pred):
    for label in h_dict:
        if m == label:
            if len(h_dict[m]) == 2:
                break
            else:
                h_dict[m].append(o)
                break

for m, o in zip(mol_cp_qm, cp_qm):
    cp_dict[m] = [o]

for m, o in zip(mol_cp_pred, cp_pred):
    for label in cp_dict:
        if m == label:
            if len(cp_dict[m]) == 2:
                break
            else:
                cp_dict[m].append(o)
                break

temperatures = np.append(298.15, range(300, 2550, 50))
nasa = []

h_t_real = []
h_t_pred = []
ae_pred = []

for mol in mol_h_qm:
    h_real = h_dict[mol][0]
    h_predicted = h_dict[mol][1]
    cp_real = cp_dict[mol][0]
    cp_predicted = cp_dict[mol][1]
    integral_real = nasa_fit(temperatures, cp_real)[1]
    integral_predicted = nasa_fit(temperatures, cp_predicted)[1]
    h_t_real.append(np.asarray(integral_real + h_real).astype(np.float))
    h_t_pred.append(np.asarray(integral_predicted + h_predicted).astype(np.float))

h_t_real = np.asarray(h_t_real).astype(np.float)
h_t_pred = np.asarray(h_t_pred).astype(np.float)
ae_pred = abs(h_t_real - h_t_pred)

np.savetxt(str(folder + "/ht_real.txt"), h_t_real, fmt='%s')
np.savetxt(str(folder + "/ht_pred.txt"), h_t_pred, fmt='%s')
np.savetxt(str(folder + "/ae_pred.txt"), ae_pred, fmt='%s')

for j in h_dict:
    print(str(j + "\t" + str(h_dict[j]) + "\t" + str(len(h_dict[j]))))
