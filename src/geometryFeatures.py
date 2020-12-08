from rdkit.Chem import rdMolTransforms, rdmolops
import math


def mendeleev(n):
    """
    Links atomic number with atom type.
    """
    periodic_table = {
        1: "H",
        2: "He",
        3: "Li",
        4: "Be",
        5: "B",
        6: "C",
        7: "N",
        8: "O",
        9: "F",
        10: "Ne",
        11: "Na",
        12: "Mg",
        13: "Al",
        14: "Si",
        15: "P",
        16: "S",
        17: "Cl",
        18: "Ar"
    }
    return periodic_table.get(n)


def bonds(conf, n, mol_h):
    """
    Returns two lists with respectively the bond lengths and the type of each bond in a molecule.
    """
    bond_values = []
    bond_names = []
    for i in range(n):
        a0 = mol_h.GetAtomWithIdx(i)
        an0 = mendeleev(a0.GetAtomicNum())
        for j in range(i+1, n):
            a1 = mol_h.GetAtomWithIdx(j)
            an1 = mendeleev(a1.GetAtomicNum())
            bond_length = rdMolTransforms.GetBondLength(conf, i, j)
            bond = "".join(sorted([an0, an1]))
            bond = cc(bond, bond_length)
            bond_values.append(bond_length)
            bond_names.append(bond)
    return bond_values, bond_names


def angles(conf, n, mol_h):
    """
    Returns two lists with respectively the angle sizes and the type of each angle in a molecule.
    """

    # conf, n, mol_h = conformer(mol)

    angle_values = []
    angle_names = []

    matrix = rdmolops.Get3DDistanceMatrix(mol_h)

    previous = []

    for i in range(n):
        a0 = mol_h.GetAtomWithIdx(i)
        an0 = mendeleev(a0.GetAtomicNum())
        for j in range(n):
            a1 = mol_h.GetAtomWithIdx(j)
            an1 = mendeleev(a1.GetAtomicNum())
            if i == j:
                continue
            for k in range(n):
                a2 = mol_h.GetAtomWithIdx(k)
                an2 = mendeleev(a2.GetAtomicNum())
                if k == i or k == j:
                    continue
                combination = sorted([i, j, k])
                cond1 = False
                cond2 = False
                cond3 = True
                # These conditions check if both atom 0 and atom 2 are connected with atom 0.
                # The atoms must be connected in this way to have physically relevant angle sizes.
                if combination in previous:
                    cond3 = False
                if matrix[i][j] < 1.65:
                    cond1 = True
                if an0 == "H" and an1 == "H":
                    cond1 = False
                if matrix[j][k] < 1.65:
                    cond2 = True
                if an1 == "H" and an2 == "H":
                    cond2 = False
                if cond1 and cond2 and cond3:
                    angle_size = rdMolTransforms.GetAngleRad(conf, i, j, k)
                    angle = "".join(sorted([an0, an1, an2]))
                    angle_values.append(angle_size)
                    angle_names.append(angle)
                    previous.append(sorted(combination))

    return angle_values, angle_names


def dihedrals(conf, n, mol_h):
    """
    Returns two lists with respectively the dihedral sizes and the type of each torsion angle in a molecule.
    Input = string
    """
    # dihedral_file = open('dihedral_types.txt', 'r')
    # dihedral_types = []
    # for line in dihedral_file:
    #     dihedral_types.append(line[:-1].split('\t'))
    # conf, n, mol_h = conformer(mol)
    dihedral_values = []
    dihedral_names = []
    matrix = rdmolops.Get3DDistanceMatrix(mol_h)
    previous = []
    for i in range(n):
        a0 = mol_h.GetAtomWithIdx(i)
        an0 = mendeleev(a0.GetAtomicNum())
        for j in range(n):
            if i == j:
                continue
            a1 = mol_h.GetAtomWithIdx(j)
            an1 = mendeleev(a1.GetAtomicNum())
            if matrix[i][j] > 1.65:
                continue
            if an0 == "H" and an1 == "H":
                continue
            for k in range(n):
                if k == i or k == j:
                    continue
                a2 = mol_h.GetAtomWithIdx(k)
                an2 = mendeleev(a2.GetAtomicNum())
                if matrix[i][k] > 1.65 and matrix[j][k] > 1.65:
                    continue
                if an0 == "H" and an2 == "H" and matrix[i][k] < 1.65:
                    continue
                if an1 == "H" and an2 == "H" and matrix[j][k] < 1.65:
                    continue
                for p in range(n):
                    if p == i or p == j or p == k:
                        continue
                    combination = sorted([i, j, k, p])
                    if combination in previous:
                        continue
                    a3 = mol_h.GetAtomWithIdx(p)
                    an3 = mendeleev(a3.GetAtomicNum())
                    if matrix[i][p] > 1.65 and matrix[j][p] > 1.65 and matrix[k][p] > 1.65:
                        continue
                    if an0 == "H" and an3 == "H" and matrix[i][p] < 1.65:
                        continue
                    if an1 == "H" and an3 == "H" and matrix[j][p] < 1.65:
                        continue
                    if an2 == "H" and an3 == "H" and matrix[k][p] < 1.65:
                        continue
                    dihedral_value = abs(rdMolTransforms.GetDihedralRad(conf, i, j, k, p))
                    if dihedral_value > math.pi/2:
                        dihedral_value = abs(math.pi - dihedral_value)
                    if str(dihedral_value).__contains__('nan'):
                        continue
                    dihedral = "".join(sorted([an0, an1, an2, an3]))
                    dihedral_values.append(dihedral_value)
                    dihedral_names.append(dihedral)
                    previous.append(combination)
    return dihedral_values, dihedral_names


def cc(bond, bond_length):
    """"
    Divides the CC bond range into five types for better coverage of physically relevant bonds
    """
    # TODO: Automate division process
    if bond == "CC":
        if bond_length < 1.25:
            bond = "C1"
        elif bond_length < 1.36:
            bond = "C2"
        elif bond_length < 1.39:
            bond = "C3"
        elif bond_length < 1.44:
            bond = "C4"
        elif bond_length < 1.46:
            bond = "CV"
        elif bond_length < 1.47:
            bond = "C5"
        elif bond_length < 1.8:
            bond = "C6"
        elif bond_length < 2.2:
            bond = "CW"
        elif bond_length < 2.7:
            bond = "C7"
        elif bond_length < 2.9:
            bond = "C8"
        elif bond_length < 4:
            bond = "C9"
        else:
            bond = "CX"
    elif bond == "CO":
        if bond_length < 1.2:
            bond = "O1"
        elif bond_length < 1.25:
            bond = "O2"
        elif bond_length < 1.4:
            bond = "OT"
        elif bond_length < 2:
            bond = "O3"
        elif bond_length < 2.27:
            bond = "O4"
        elif bond_length < 2.5:
            bond = "O5"
        elif bond_length < 3.9:
            bond = "O6"
        else:
            bond = "O7"
    return bond
