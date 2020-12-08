from src.geometryFeatures import bonds, angles, dihedrals
import matplotlib.pyplot as plt
from src.makeMolecule import conformer, input_type, molecule
from rdkit import Chem
from rdkit.Chem import AllChem


def all_values(data):
    hist_input = {}
    bad = []
    for mol in data:
        print(mol)
        if input_type(mol):
            try:
                conf, n, mol_h = conformer(mol)
                # TODO: More efficient way
            except ValueError:
                try:
                    conf, n, mol_h = conformer(mol)
                except ValueError:
                    m = molecule(mol)
                    mol_h = Chem.AddHs(m)
                    try:
                        AllChem.EmbedMolecule(mol_h, useRandomCoords=True)
                        AllChem.UFFOptimizeMolecule(mol_h)
                        conf, n, mol_h = mol_h.GetConformer(), mol_h.GetNumAtoms(), mol_h
                    except ValueError:
                        print("{} is a bad molecule".format(mol))
                        bad.append(list(data).index(mol))
                        continue
            dist = bonds(conf, n, mol_h)
            angs = angles(conf, n, mol_h)
            dihs = dihedrals(conf, n, mol_h)
            geo = (dist[0] + angs[0] + dihs[0], dist[1] + angs[1] + dihs[1])
        # else:
        #     mol_h = conformer(mol)
        #     dist = bonds_ob(mol_h)
        #     angs = angles_ob(mol_h)
        #     dihs = dihedrals_ob(mol_h)
        #     geo = (dist[0] + angs[0] + dihs[0], dist[1] + angs[1] + dihs[1])
        for value, name in zip(*geo):
            # print(value, name)
            if name in hist_input:
                hist_input[name].append(value)
            else:
                hist_input[name] = [value]
    return hist_input, bad


def histogram_plot(data, geometry_type):
    values = all_values(data)[geometry_type]
    num_bins = 200
    plt.hist(values, num_bins, facecolor='blue', alpha=0.5)
    if len(geometry_type) == 2:
        plt.xlabel("Distance [Ã…]")
    else:
        plt.xlabel("Angle [rad]")
    plt.ylabel("Occurrence")
    plt.title(geometry_type)
    plt.savefig(str(geometry_type + ".png"))
