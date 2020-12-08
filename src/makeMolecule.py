from rdkit import Chem
from rdkit.Chem import AllChem, rdmolfiles, Descriptors
from rdkit.Chem.rdMolTransforms import ComputePrincipalAxesAndMoments as Inertia
from rdkit.Chem.Descriptors import NumRadicalElectrons
from openbabel import pybel
import numpy as np
import math


def molecule_test_list(molecule_file):
    """
    This function turns the input file into a list of identifiers.
    Similar to molecule_list, without outputs provided.
    """

    with open(molecule_file, 'r') as f:
        molecules_full = f.readlines()

    molecules = []

    for line in molecules_full:
        line = line[:-1]
        molecules.append(line)

    if len(molecules) == 1:
        print("{} contains {} molecule".format(str(molecule_file), len(molecules)))
    else:
        print("{} contains {} molecules".format(str(molecule_file), len(molecules)))

    return molecules


def molecule_list(molecule_file, suppress="no"):
    """
    This function turns the input file into a list of identifiers.
    """

    with open(molecule_file, 'r') as f:
        molecules_full = f.readlines()

    molecules = []  # Create empty lists
    outputs = []
    bad_molecules = []

    for line in molecules_full:
        line = line[:-1].split('\t')

        # Check if the molecule can be handled by RDKit
        # TODO: The parsing step takes too much time ==> Optimize!

        try:
            conformer(line[0])
        except ValueError:
            try:
                conformer(line[0])
            except ValueError:
                print("{} is a bad molecule!".format(line[0]))
                bad_molecules.append(line[0])
                continue

        # TODO: Instead of removing the molecule, a method must be created to figure out the coordinates of the molecule

        print(line[0])
        molecules.append(line[0])

        # Read the outputs. Different outputs are distinguished:
        #  1) More than value is predicted (e.g. c_p)
        #  2) Values are separated by spaces instead of tabs
        #  3) Regular input files

        if len(line) > 2:
            outputs.append(line[1:])
        else:
            if len(line[0].split(' ')) > 1:
                line = line[0].split(' ')
                outputs.append(line[1:])
            else:
                outputs.append(line[1])

    # Suppresses the printed output about the file length

    if suppress == "no":
        if len(molecules) == 1:
            print("{} contains {} molecule".format(str(molecule_file), len(molecules)))
        else:
            print("{} contains {} molecules".format(str(molecule_file), len(molecules)))

    return molecules, outputs, bad_molecules


def molecule(line):
    """
    This functions returns an RDKit molecule object.
    """
    if line.__contains__("InChI"):  # Molecule in InChI format
        return Chem.MolFromInchi(line)
    elif ":" in line:  # Molecule as parsed.mol file
        mol_file = str(line + "/parsed.mol")
        mol = next(pybel.readfile("mol", mol_file))
        mol = mol.OBMol
        n = mol.NumAtoms()
        xv = []
        yv = []
        zv = []
        for i in range(n):
            xv.append(mol.GetAtomById(i).GetX())
            yv.append(mol.GetAtomById(i).GetY())
            zv.append(mol.GetAtomById(i).GetZ())
        xyz = list(zip(xv, yv, zv))
        if str(xyz).__contains__('nan'):
            print("The coordinates of {} are incorrect!".format(mol_file))
            raise ValueError
        return mol
    else:  # Molecule as SMILES
        return Chem.MolFromSmiles(line)


def input_type(line):
    if line.__contains__("InChI"):
        return True
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
        # for c in mol.GetConformers():
        #     print(c.GetPositions())
        return mol
    else:
        mol = molecule(mol_name)
        mol_h = Chem.AddHs(mol)
        AllChem.EmbedMolecule(mol_h, useRandomCoords=True)
        try:
            AllChem.MMFFOptimizeMolecule(mol_h)
        except ValueError:
            try:
                AllChem.EmbedMolecule(mol_h, useRandomCoords=True)
                AllChem.UFFOptimizeMolecule(mol_h)
            except ValueError:
                pass
        return mol_h.GetConformer(), mol_h.GetNumAtoms(), mol_h


def heavy_atoms(mol):
    m = molecule(mol)
    ha = m.GetNumHeavyAtoms()
    return ha


def normalize(molecules, outputs, thermo, coefficient=None):
    property_dict = {"entropy": 1,
                     "s": 1,
                     "cp": 1.5}
    if coefficient is None:
        coefficient = property_dict[thermo]
    normalized_output = []
    heavy = []
    for mol, output in zip(molecules, outputs):
        ha = heavy_atoms(mol)
        heavy.append(ha)
        normed = output / (math.log10(ha) ** coefficient)
        normalized_output.append(normed)
    normalized_output = np.asarray(normalized_output).astype(np.float)
    return normalized_output, heavy


def denormalize(outputs, heavy, thermo, coefficient=None):
    property_dict = {"entropy": 1,
                     "s": 1,
                     "cp": 1.5}
    if coefficient is None:
        coefficient = property_dict[thermo]
    else:
        coefficient = float(coefficient)
    original = []
    for s, n in zip(outputs, heavy):
        original.append(s * (math.log10(n)) ** coefficient)
    original = np.asarray(original).astype(np.float)
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
