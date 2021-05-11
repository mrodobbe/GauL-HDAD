from src.makeMolecule import molecule
from rdkit import Chem
from rdkit.Chem import rdDistGeom, AllChem, rdMolDescriptors


def conformer_generation(mol_name):
    mol = Chem.AddHs(molecule(mol_name))
    print(mol_name)
    n_atoms = mol.GetNumAtoms()
    n = rdMolDescriptors.CalcNumRotatableBonds(mol)
    num_confs = min(3 ** n, 243)
    rdDistGeom.EmbedMultipleConfs(mol, num_confs, rdDistGeom.ETKDGv3())
    AllChem.MMFFOptimizeMoleculeConfs(mol, mmffVariant='MMFF94s')
    confs = mol.GetConformers()
    return confs, n_atoms, mol
