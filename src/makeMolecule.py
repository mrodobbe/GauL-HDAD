from rdkit import Chem
from rdkit.Chem import AllChem, rdmolfiles, Descriptors
from rdkit.Chem.rdMolTransforms import ComputePrincipalAxesAndMoments as Inertia
from rdkit.Chem.Descriptors import NumRadicalElectrons
import numpy as np
import math
import os


def molecule_test_list(molecule_file):
    """
    This function turns the input file into a list of identifiers.
    Similar to molecule_list, without outputs provided.
    """

    license_disclaimer()
    try:
        with open(molecule_file, 'r') as f:
            molecules_full = f.readlines()
    except FileNotFoundError:
        print("The input file does not exist.")
        raise

    molecules = []
    conformers = []
    bad_molecules = []

    for line in molecules_full:
        line = line[:-1]
        try:
            c = conformer(line)
            conformers.append(c)
        except ValueError:
            print("{} is a bad molecule!".format(line[0]))
            bad_molecules.append(line[0])
            continue

        # TODO: Instead of removing the molecule, a method must be created to figure out the coordinates of the molecule

        # print(line[0])
        molecules.append(line)

    if len(molecules) == 1:
        print("{} contains {} molecule".format(str(molecule_file), len(molecules)))
    else:
        print("{} contains {} molecules".format(str(molecule_file), len(molecules)))

    return molecules, conformers, bad_molecules


def molecule_list(molecule_file, suppress="no"):
    """
    This function turns the input file into a list of identifiers.
    """
    license_disclaimer()
    try:
        with open(molecule_file, 'r') as f:
            molecules_full = f.readlines()
    except FileNotFoundError:
        print("The input file does not exist.")
        raise

    molecules = []  # Create empty lists
    outputs = []
    conformers = []
    bad_molecules = []

    for line in molecules_full:
        line = line[:-1].split('\t')
        try:
            c = conformer(line[0])
            conformers.append(c)
        except ValueError:
            print("{} is a bad molecule!".format(line[0]))
            bad_molecules.append(line[0])
            continue

        # TODO: Instead of removing the molecule, a method must be created to figure out the coordinates of the molecule

        # print(line[0])
        molecules.append(line[0])

        # Read the outputs. Different outputs are distinguished:
        #  1) More than value is predicted (e.g. c_p)
        #  2) Values are separated by spaces instead of tabs
        #  3) Regular input files

        if len(line) > 2:
            outputs.append(line[1:])
        else:
            if len(line) == 1 and len(line[0].split(' ')) > 1:
                line = line[0].split(' ')
                outputs.append(line[1:])
            else:
                outputs.append(line[1])

    bad_molecules = np.asarray(bad_molecules)
    outputs = np.asarray(outputs).astype(np.float)

    # Suppresses the printed output about the file length

    if suppress == "no":
        if len(molecules) == 1:
            print("{} contains {} molecule".format(str(molecule_file), len(molecules)))
        else:
            print("{} contains {} molecules".format(str(molecule_file), len(molecules)))

    return molecules, outputs, conformers, bad_molecules


def store_bad_molecules(bad_molecules, save_folder):
    if len(bad_molecules) > 0:
        np.savetxt(str(save_folder + "/bad_molecules.txt"), bad_molecules, fmt="%s")
        print("Dumped a list with molecules which could not be parsed in Output/bad_molecules.txt")
    else:
        print("All molecules can be parsed by RDKit")


def molecule(line):
    """
    This functions returns an RDKit molecule object.
    """
    if line.__contains__("InChI"):  # Molecule in InChI format
        return Chem.MolFromInchi(line)
    elif line.endswith(".mol"):
        return rdmolfiles.MolFromMolFile(line, removeHs=False)
    elif ":" in line:  # Molecule as parsed.mol file
        if line.endswith(".xyz"):
            mol_file = str(line[:-8] + "/parsed.mol")
        elif line.endswith(".mol"):
            mol_file = line
        elif line[-3] == "." or line[-4] == ".":
            print("Unknown file format.\nPlease make for each molecule a folder with a parsed.mol file.")
            raise NameError
        else:
            mol_file = str(line + "/parsed.mol")
        return rdmolfiles.MolFromMolFile(mol_file, removeHs=False)
    else:  # Molecule as SMILES
        return Chem.MolFromSmiles(line)


def input_type(line):
    if line.__contains__("InChI"):
        return True
    elif line.__contains__(".mol"):
        return False
    elif ":" in line:
        return False
    else:
        return True


def conformer(mol_name):
    """
    Create a three-dimensional structure from an RDKit molecule object.
    Two variables are returned:
    1) An RDKit conformer structure
    2) The number of heavy atoms
    """
    if not input_type(mol_name):
        mol = molecule(mol_name)
        conf = mol.GetConformer()
        n = mol.GetNumAtoms()
        return conf, n, mol
    else:
        mol = molecule(mol_name)
        mol_h = Chem.AddHs(mol)
        AllChem.EmbedMolecule(mol_h, useRandomCoords=True)
        try:
            AllChem.MMFFOptimizeMolecule(mol_h, maxIters=10000)
        except ValueError:
            pass
        return mol_h.GetConformer(), mol_h.GetNumAtoms(), mol_h


def heavy_atoms(mol):
    m = molecule(mol)
    ha = m.GetNumHeavyAtoms()
    return ha


def normalize(molecules, outputs, thermo, coefficient=None):
    property_dict = {"entropy": 1.5,
                     "s": 1.5,
                     "cp": 1.5}
    if coefficient is None and thermo in property_dict:
        coefficient = property_dict[thermo]
    elif coefficient is not None:
        coefficient = float(coefficient)
    if thermo in property_dict or coefficient is not None:
        normalized_output = []
        heavy = []
        for mol, output in zip(molecules, outputs):
            ha = heavy_atoms(mol)
            heavy.append(ha)
            normed = output / (math.log(ha) ** coefficient)
            normalized_output.append(normed)
    else:
        heavy = [heavy_atoms(mol) for mol in molecules]
        normalized_output = outputs
    normalized_output = np.asarray(normalized_output).astype(np.float)
    heavy = np.asarray(heavy)
    return normalized_output, heavy


def denormalize(outputs, heavy, thermo, coefficient=None):
    property_dict = {"entropy": 1.5,
                     "s": 1.5,
                     "cp": 1.5}
    if coefficient is None and thermo in property_dict:
        coefficient = property_dict[thermo]
    elif coefficient is not None:
        coefficient = float(coefficient)
    if thermo in property_dict or coefficient is not None:
        original = []
        for s, n in zip(outputs, heavy):
            original.append(s * (math.log(n)) ** coefficient)
        original = np.asarray(original).astype(np.float)
    else:
        original = outputs
    return original


def nasa_fit(temperatures, values):
    z = np.polyfit(temperatures, values, 4)
    p = np.poly1d(z)
    integrand = p.integ()
    integrated = np.asarray([(integrand(t) - integrand(min(temperatures)))/1000 for t in temperatures])
    return z, integrated


def add_inertia(molecules, representations):
    t_r = []
    for mol, representation in zip(molecules, representations):
        conf = conformer(mol)[0]
        im_xyz = np.asarray(Inertia(conf)[1]).astype(np.float)
        for i in im_xyz:
            if i < 1e-6:
                i = 0.0
        im = (im_xyz[0] * im_xyz[1] * im_xyz[2]) ** (1 / 3)
        if im < 1e-4:
            im_log = 0
        else:
            im_log = math.log10(im)
        if np.isnan(im_log):
            im_log = 0
        t = np.append(representation, im_log)
        t_r.append(t)
    new_representations = np.stack(t_r)
    return new_representations


def add_radical(molecules, representations):
    t_r = []
    for mol, representation in zip(molecules, representations):
        m = molecule(mol)
        rad = NumRadicalElectrons(m)
        t = np.append(representation, rad)
        t_r.append(t)
    new_representations = np.stack(t_r)
    return new_representations


def add_rings(molecules, representations):
    t_r = []
    for mol, representation in zip(molecules, representations):
        m = molecule(mol)
        ri = m.GetRingInfo()
        num_rings = ri.AtomRings()
        one_hot = np.zeros(8)
        print(mol)
        for i in range(len(num_rings)):
            if len(num_rings[i]) > 10:
                one_hot[-1] += 1
            else:
                one_hot[len(num_rings[i])-3] += 1
        t = np.append(representation, one_hot)
        t_r.append(t)
    new_representations = np.stack(t_r)
    return new_representations


def add_mw(molecules, representations):
    t_r = []
    for mol, representation in zip(molecules, representations):
        m = molecule(mol)
        mw = Descriptors.MolWt(m)
        t = np.append(representation, mw)
        t_r.append(t)
    new_representations = np.stack(t_r)
    return new_representations


def add_n(molecules, representations):
    t_r = []
    for mol, representation in zip(molecules, representations):
        m = molecule(mol)
        n = m.GetNumAtoms()
        t = np.append(representation, n)
        t_r.append(t)
    new_representations = np.stack(t_r)
    return new_representations


def input_checker(args, filename):
    if len(args) < 4:
        print("Not enough input files.\nPlease use the following command structure:\n"
              "python {}.py <molecule_file> <property> <save_folder>".format(filename))
        raise IndexError
    else:
        if filename == "train":
            folder = args[3]
            try:
                os.mkdir(folder)
            except PermissionError:
                print("No permission to create folder.\nThere are two ways to solve the problem:\n"
                      "1) Open Anaconda as Administrator\n"
                      "2) Manually create the subdirectory Output with 2 additional "
                      "subdirectories in Output: gmm and hist")
                raise
            except FileExistsError:
                print("Folder already exists. Data in this folder will be overwritten.")

            try:
                os.mkdir(str(folder + "/gmm"))
                os.mkdir(str(folder + "/hist"))
            except FileExistsError:
                print("Folders already exist.")


def license_disclaimer():
    print("\n\n------------------------------------------------------------------------------------------------------\n"
          "GauL-HDAD  Copyright (C) 2021 Maarten R. Dobbelaere \n"
          "This program comes with ABSOLUTELY NO WARRANTY; for details open the `LICENSE` file. \n"
          "This is free software, and you are welcome to redistribute it \n"
          "under certain conditions; for details open the `LICENSE` file.\n\n"
          "When using GauL-HDAD for your own publication, please refer to the original paper:\n"
          "Learning Molecular Representations for Thermochemistry Prediction of Cyclic Hydrocarbons and Oxygenates\n"
          "Dobbelaere, M.R.; Plehiers, P.P.; Van de Vijver, R.; Stevens, C.V.; Van Geem, K.M.\n"
          "Submitted to Journal of Physical Chemistry A, 2021\n"
          "------------------------------------------------------------------------------------------------------\n\n")
