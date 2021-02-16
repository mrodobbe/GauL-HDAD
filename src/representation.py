import numpy as np
from src.geometryFeatures import bonds, angles, dihedrals
from src.gaussian import gauss
from src.makeMolecule import conformer, input_type, add_radical


def gaul_representation(mol, conformer_tuple, theta_dict):
    print(mol)
    representation_dict = {}
    conf, n, mol_h = conformer_tuple
    dist = bonds(conf, n, mol_h)
    angs = angles(conf, n, mol_h)
    dihs = dihedrals(conf, n, mol_h)
    geo = (dist[0] + angs[0] + dihs[0], dist[1] + angs[1] + dihs[1])
    for key in theta_dict:
        q = theta_dict[key]
        t = np.asarray(q)
        if len(t.shape) > 1:
            representation_dict[key] = np.zeros(t.shape[0])
        else:
            representation_dict[key] = [0]
    for value, name in zip(*geo):
        g = []
        for key in theta_dict:
            if name == key:
                theta = np.asarray(theta_dict[key])
                break
        try:
            if len(theta.shape) > 1:
                for mu, sig in theta:
                    gd = gauss(value, mu, sig)
                    g.append(gd)
                gs = sum(g)
                if gs == 0:
                    continue
                else:
                    gt = g/gs
            # elif name == "C1":
            #     gt = [1]
            # elif name == "O1":
            #     gt = [1]
            else:
                gd = gauss(value, theta[0], theta[1])
                g.append(gd)
        except UnboundLocalError:
            continue
        for feat in representation_dict:
            if name == feat:
                representation_dict[feat] = np.add(representation_dict[feat], gt)
    r = []
    for part in representation_dict:
        p = np.asarray(representation_dict[part])
        r = np.append(r, p)
    r = np.asarray(r).astype(np.float)
    return r


def represent(molecules, conformers, gmm_dict):
    representations = []
    bad = []
    molecules = list(molecules)
    print("Start representing the molecules!")
    for mol, conformer_tuple in zip(molecules, conformers):
        try:
            v = gaul_representation(mol, conformer_tuple, gmm_dict)
        except ValueError:
            print("Bad molecule at index {}".format(molecules.index(mol)))
            bad.append(molecules.index(mol))
            continue
        r = np.asarray(v)
        representations.append(r)
    stacked_representations = np.stack(representations)
    stacked_representations = add_radical(molecules, stacked_representations)
    print("Finished representing the molecules")
    return stacked_representations, bad
