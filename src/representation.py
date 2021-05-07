import numpy as np
import pickle
from src.geometryFeatures import bonds, angles, dihedrals
from src.gaussian import gauss
from src.makeMolecule import add_radical


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
            elif name == "C1":
                gt = [1]
            # elif name == "O1":
            #     gt = [1]
            elif theta.shape[0] == 0:
                continue
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


def represent(molecules, conformers, gmm_dict, save_folder):
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
    with open(str(save_folder + "/representations.pickle"), "wb") as f:
        pickle.dump(representations, f)
    print("Dumped the molecule representations!")
    # text_representation(molecules, stacked_representations, gmm_dict, save_folder)
    return stacked_representations, bad


def load_representations(molecules, conformers, save_folder):
    try:
        with open(str(save_folder + "/test_representations.pickle"), "rb") as f:
            representations = pickle.load(f)
        print("Loaded the molecule representations!")
        representations = np.asarray(representations).astype(np.float)
        return representations, molecules
    except FileNotFoundError:
        print("No representations available! Trying to find a gmm dictionary")
        try:
            with open(str(save_folder + "/gmm_dictionary.pickle"), "rb") as f:
                gmm_dictionary = pickle.load(f)
            print("Loaded the GMM data!")
            representations, bad = represent(molecules, conformers, gmm_dictionary, save_folder)
            molecules = np.delete(molecules, bad)
            return representations, molecules
        except FileNotFoundError:
            print("No gmm dictionary found. Please include a gmm dictionary in {}".format(save_folder))
            raise


def text_representation(molecules, representations, gmm_dict, save_folder):
    labels = ["Label"]
    mu = ["Mu"]
    sigma = ["Sigma"]
    for label in gmm_dict:
        for m, s in gmm_dict[label]:
            labels.append(label)
            mu.append(round(m, 4))
            sigma.append(round(s, 4))
    labels.append("Radical")
    mu.append("0")
    sigma.append("0")
    with_name = np.vstack((molecules, representations.T)).T
    print(with_name.shape)
    print(len(labels))
    all_data = np.vstack((labels, mu, sigma, with_name))
    np.savetxt(str(save_folder + "/fingerprints.txt"), all_data, fmt='%s')
    print("The molecular fingerprints can be evaluated in {}".format(str(save_folder + "/fingerprints.txt")))
