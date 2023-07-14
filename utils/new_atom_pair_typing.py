from typing import List, Tuple, Union, Dict, Optional
import math
import os
import pandas as pd 
from tqdm import tqdm
import pickle
import csv
import cupy as cp
import numpy as np
import torch
import torch.nn as nn
import rdkit
from rdkit import Chem
from rdkit.Chem import AllChem
from rdkit.Chem import QED, MACCSkeys
from sklearn.metrics import pairwise_distances
import pickle
from collections import defaultdict

MAX_ATOMIC_NUM = 100

ATOM_FEATURES = {
    'atomic_num': list(range(MAX_ATOMIC_NUM)),
    'degree': [0, 1, 2, 3, 4, 5],
    'formal_charge': [-1, -2, 1, 2, 0],
    'chiral_tag': [0, 1, 2, 3],
    'num_Hs': [0, 1, 2, 3, 4],
    'hybridization': [
        Chem.rdchem.HybridizationType.SP,
        Chem.rdchem.HybridizationType.SP2,
        Chem.rdchem.HybridizationType.SP3,
        Chem.rdchem.HybridizationType.SP3D,
        Chem.rdchem.HybridizationType.SP3D2
    ],
}

cur_atom_typing = {
    'atomic_symbol': ['C', 'H', 'O', 'N', 'S', 'P', 'F', 'Cl', 'Br', 'I'],
    'degree': [0, 1, 2, 3, 4],
    'formal_charge': [-1, -2, 1, 2, 0],
    'chiral_tag': [0, 1, 2, 3],
    'num_Hs': [0, 1, 2, 3, 4],
    'hybridization': [
        Chem.rdchem.HybridizationType.SP,
        Chem.rdchem.HybridizationType.SP2,
        Chem.rdchem.HybridizationType.SP3,
        Chem.rdchem.HybridizationType.SP3D,
        Chem.rdchem.HybridizationType.SP3D2
    ],
}

ATOM_FDIM = sum(len(choices) + 1 for choices in ATOM_FEATURES.values()) + 2
BOND_FDIM = 14
# cur_node_num = 0
def count_atom_typing(path):
    atom_types_list = []
    hybrid_types_list = []
    degree_types_list = []
    formal_charge_types_list = []
    chrial_tag_types_list = []
    num_Hs_types_list = []
    atom_types_count_dict = defaultdict(int)
    hybrid_type_count_dict = defaultdict(int)
    degree_type_count_dict = defaultdict(int)
    formal_charge_count_dict = defaultdict(int)
    chrial_tag_count_dict = defaultdict(int)
    num_Hs_count_dict = defaultdict(int)

    for filename in path:
        result = pd.read_csv(filename).values.tolist()
        for context in result:
            mol = Chem.MolFromSmiles(context[0])
            atoms = mol.GetAtoms()
            atoms_num = mol.GetNumAtoms()
            one_ligand_list = []
            coord_list = []
            if atoms_num <= 100:
                for i, at in enumerate(atoms):
                    idx, num, symbol, hybrid, aromatic, degree, formal_charge, chrial_tag, num_Hs = at.GetIdx(), at.GetAtomicNum(), at.GetSymbol(), at.GetHybridization(), int(at.GetIsAromatic()), at.GetTotalDegree(), at.GetFormalCharge(), int(at.GetChiralTag()), int(at.GetTotalNumHs())
                    atom_types_count_dict[symbol] += 1
                    hybrid_type_count_dict[hybrid] += 1
                    degree_type_count_dict[degree] += 1
                    formal_charge_count_dict[formal_charge] += 1
                    chrial_tag_count_dict[chrial_tag] += 1
                    num_Hs_count_dict[num_Hs] += 1

    for key, val in atom_types_count_dict.items():
        if val >= 100:
            atom_types_list.append(key)
    cur_atom_typing['atomic_symbol'] = atom_types_list
    for key, val in hybrid_type_count_dict.items():
        if val >= 100:
            hybrid_types_list.append(key)
    cur_atom_typing['hybridization'] = hybrid_types_list
    for key, val in degree_type_count_dict.items():
        if val >= 100:
            degree_types_list.append(key)
    cur_atom_typing['degree'] = degree_types_list
    for key, val in formal_charge_count_dict.items():
        if val >= 10:
            formal_charge_types_list.append(key)
    cur_atom_typing['formal_charge'] = formal_charge_types_list
    for key, val in chrial_tag_count_dict.items():
        if val >= 10:
            chrial_tag_types_list.append(key)
    cur_atom_typing['chiral_tag'] = chrial_tag_types_list
    for key, val in num_Hs_count_dict.items():
        if val >= 10:
            num_Hs_types_list.append(key)
    cur_atom_typing['num_Hs'] = num_Hs_types_list

def onek_encoding_unk(value: int, choices: List[int]) -> List[int]:
    """
    Creates a one-hot encoding.

    :param value: The value for which the encoding should be one.
    :param choices: A list of possible values.
    :return: A one-hot encoding of the value in a list of length len(choices) + 1.
    If value is not in the list of choices, then the final element in the encoding is 1.
    """
    encoding = [0] * (len(choices) + 1)
    index = choices.index(value) if value in choices else -1
    encoding[index] = 1

    return encoding

def atom_features(atom: Chem.rdchem.Atom) -> List[Union[bool, int, float]]:
    """
    Builds a feature vector for an atom.

    :param atom: An RDKit atom.
    :param functional_groups: A k-hot vector indicating the functional groups the atom belongs to.
    :return: A list containing the atom features.
    """
    features = onek_encoding_unk(atom.GetAtomicNum() - 1, ATOM_FEATURES['atomic_num']) + \
           onek_encoding_unk(atom.GetTotalDegree(), ATOM_FEATURES['degree']) + \
           onek_encoding_unk(atom.GetFormalCharge(), ATOM_FEATURES['formal_charge']) + \
           onek_encoding_unk(int(atom.GetChiralTag()), ATOM_FEATURES['chiral_tag']) + \
           onek_encoding_unk(int(atom.GetTotalNumHs()), ATOM_FEATURES['num_Hs']) + \
           onek_encoding_unk(int(atom.GetHybridization()), ATOM_FEATURES['hybridization']) + \
           [1 if atom.GetIsAromatic() else 0] + \
           [atom.GetMass() * 0.01]  # scaled to about the same range as other features
    
    return features

def atom_typing_encoding(atom: Chem.rdchem.Atom) -> List[Union[bool, int, float]]:    
    
    atom_typing = onek_encoding_unk(atom.GetSymbol(), cur_atom_typing['atomic_symbol']) + \
           onek_encoding_unk(atom.GetTotalDegree(), cur_atom_typing['degree']) + \
           onek_encoding_unk(atom.GetFormalCharge(), cur_atom_typing['formal_charge']) + \
           onek_encoding_unk(int(atom.GetChiralTag()), cur_atom_typing['chiral_tag']) + \
           onek_encoding_unk(int(atom.GetTotalNumHs()), cur_atom_typing['num_Hs']) + \
           onek_encoding_unk(int(atom.GetHybridization()), cur_atom_typing['hybridization']) + \
           [1 if atom.GetIsAromatic() else 0] + \
           [atom.GetMass() * 0.01] 
    
    return atom_typing


def bond_features(bond: Chem.rdchem.Bond) -> List[Union[bool, int, float]]:
    """
    Builds a feature vector for a bond.

    :param bond: A RDKit bond.
    :return: A list containing the bond features.
    """
    if bond is None:
        fbond = [1] + [0] * (BOND_FDIM - 1)
    else:
        bt = bond.GetBondType()
        fbond = [
            0,  # bond is not None
            bt == Chem.rdchem.BondType.SINGLE,
            bt == Chem.rdchem.BondType.DOUBLE,
            bt == Chem.rdchem.BondType.TRIPLE,
            bt == Chem.rdchem.BondType.AROMATIC,
            (bond.GetIsConjugated() if bt is not None else 0),
            (bond.IsInRing() if bt is not None else 0)
        ]
        fbond += onek_encoding_unk(int(bond.GetStereo()), list(range(6)))
    return fbond

def startwith(start: int, mgraph: list) -> list:
    passed = [start]
    nopass = [x for x in range(len(mgraph)) if x != start]
    dis = mgraph[start]
    
    while len(nopass):
        idx = nopass[0]
        for i in nopass:
            if dis[i] < dis[idx]: idx = i

        nopass.remove(idx)
        passed.append(idx)

        for i in nopass:
            if dis[idx] + mgraph[idx][i] < dis[i]: dis[i] = dis[idx] + mgraph[idx][i]
    return dis   

def do_pair_encoding(args, path, fold, filename, set_name):
    
    # # Count the number of atom typing in a particular dataset
    global MAX_ATOMIC_NUM
    # if fold == 0:
    #     count_atom_typing(path)
    cur_node_num = sum(len(choices) + 1 for choices in cur_atom_typing.values()) + 2
    print(f'The dataset now processed is {filename}')
    result = pd.read_csv(filename).values.tolist()
    ligand_graph_list = []
    ligand_nodes_list = []
    dij_matrix = []
    f_atoms_list = []
    f_bonds_list = []
    a2b_list = []
    b2a_list = []
    b2revb_list = []
    bonds_list = []
    Y = []
    smi = []
    adj_list = []
    cur_result = []
    for context in result:
        mol = Chem.MolFromSmiles(context[0])
        if mol:
            cur_result.append(context)
    for context in cur_result:
        mol = Chem.MolFromSmiles(context[0])
        if args.task in ['qm7','qm8','qm9']:
            xyz = eval(context[1])
            MAX_ATOMIC_NUM = len(xyz)
        n_atoms = mol.GetNumAtoms()
        m = Chem.AddHs(mol)
        atoms = m.GetAtoms()
        atoms_num = m.GetNumAtoms()
        n_bonds = 0
        one_ligand_list = []
        f_atoms = []
        f_bonds = []
        a2b = []
        b2a = []
        b2revb = []
        bonds = []
        if atoms_num <= 100:
            for i, atom in enumerate(mol.GetAtoms()):
                f_atoms.append(atom_features(atom))
            for i, at in enumerate(atoms):
                one_ligand_list.append(atom_typing_encoding(at))

            # atom features
            f_atoms = [f_atoms[i] for i in range(n_atoms)]

            adj = np.zeros((n_atoms, n_atoms))
            inf = 9999
            for a1 in range(n_atoms):
                for a2 in range(n_atoms):
                    bond = mol.GetBondBetweenAtoms(a1, a2)
                    if bond:
                        adj[a1][a2] = 1
                    else:
                        if a1!=a2:
                            adj[a1][a2] = inf
            for i in range(n_atoms):
                dis = startwith(i, adj)
            
            padded_array = np.zeros((MAX_ATOMIC_NUM, MAX_ATOMIC_NUM))
            padded_array[:adj.shape[0], :adj.shape[1]] = adj

            for _ in range(n_atoms):
                a2b.append([])

            # bond features
            for a1 in range(n_atoms):
                for a2 in range(a1 + 1, n_atoms):
                    bond = mol.GetBondBetweenAtoms(a1, a2)

                    if bond is None:
                        continue

                    f_bond = bond_features(bond)

                    f_bonds.append(f_atoms[a1] + f_bond)
                    f_bonds.append(f_atoms[a2] + f_bond)

                    b1 = n_bonds
                    b2 = b1 + 1
                    a2b[a2].append(b1)  
                    b2a.append(a1)
                    a2b[a1].append(b2) 
                    b2a.append(a2)
                    b2revb.append(b2)
                    b2revb.append(b1)
                    n_bonds += 2
                    bonds.append(np.array([a1, a2]))
            one_ligand_list = one_ligand_list + (MAX_ATOMIC_NUM - len(one_ligand_list)) * [[0] * cur_node_num]
            ligand_nodes_list.append(one_ligand_list)
            f_atoms_list.append(f_atoms)
            f_bonds_list.append(f_bonds)
            a2b_list.append(a2b)
            b2a_list.append(b2a)
            b2revb_list.append(b2revb)
            bonds_list.append(bonds)
            if args.task in ['qm7','qm8','qm9']:
                dij = pairwise_distances(np.array(xyz), np.array(xyz))
                dij_matrix.append(dij)
                label = [float(x) if x != '' else None for x in context[3:]]
            else:
                label = [float(x) if x != '' else None for x in context[1:]]
            Y.append(label)
            adj_list.append(padded_array)
    
    os.makedirs(f"{args.saved_dir}/{args.task}/", exist_ok=True)
    
    np.save(f"{set_name}-Y", np.array(Y))

    ligand_x = np.array(ligand_nodes_list)
    pocket_x = np.array(ligand_nodes_list)
    dist_x = np.array(dij_matrix)
    adj_x = np.array(adj_list)
    print(ligand_x.shape)

    os.makedirs(f"{set_name}", exist_ok=True)

    for index in tqdm(range(ligand_x.shape[0])):
        molgraph = {}
        ligand_one_hot = ligand_x[index]
        pocket_one_hot = pocket_x[index]
        f_atoms = f_atoms_list[index]
        f_bonds = f_bonds_list[index]
        a2b = a2b_list[index]
        b2a = b2a_list[index]
        b2revb = b2revb_list[index]
        bonds = bonds_list[index]
        pocket = np.expand_dims(pocket_one_hot, axis=1)
        ligand = np.expand_dims(ligand_one_hot, axis=0)
        pocket_encoding = np.tile(pocket, (1, MAX_ATOMIC_NUM, 1))
        ligand_encoding = np.tile(ligand, (MAX_ATOMIC_NUM, 1, 1))
        ligand_pair_encoding = np.tile(ligand, (MAX_ATOMIC_NUM, 1, 1))
        l_l_x = np.concatenate((ligand_pair_encoding, ligand_pair_encoding), axis=2)
        T_ij = np.concatenate((pocket_encoding, ligand_encoding), axis=2)
        d_ij = np.expand_dims(adj_x[index], axis=2)
        molgraph['pair_one_coding'] = T_ij
        molgraph['l_l_x'] = l_l_x
        molgraph['d_ij'] = d_ij
        if args.task in ['qm7','qm8','qm9']:
            dij_matrix_np = np.expand_dims(dist_x[index], axis=2)
            molgraph['d_ij'] = dij_matrix_np
        molgraph['f_atoms'] = f_atoms
        molgraph['f_bonds'] = f_bonds
        molgraph['a2b'] = a2b
        molgraph['b2a'] = b2a
        molgraph['b2revb'] = b2revb
        molgraph['bonds'] = bonds
        with open(f"{set_name}/mol_graph_{index}.pkl", 'wb') as handle:
            pickle.dump(molgraph, handle, protocol=pickle.HIGHEST_PROTOCOL)
