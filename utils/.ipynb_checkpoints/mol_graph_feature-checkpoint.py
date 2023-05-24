import pandas as pd
from collections import OrderedDict
from typing import List, Tuple, Union, Dict, Optional

from rdkit import Chem
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import math
import pickle

BOND_FDIM = 14
SMILES_TO_GRAPH = {}
# Atom feature sizes
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

ATOM_FDIM = sum(len(choices) + 1 for choices in ATOM_FEATURES.values()) + 2
    
def load_smi(path):
    import pickle
    with open(path, 'rb') as f:
        smi = pickle.load(f)
    return smi


def get_mol_graph_fea(index, x_path):
    mol_graphs = []
    for i in index.tolist():
        with open(f"{x_path}/mol_graph_{int(i)}.pkl", 'rb') as handle:
            molgraph = pickle.load(handle)

        mol_graphs.append(molgraph)
    
    atom_fdim = ATOM_FDIM
    bond_fdim = ATOM_FDIM + BOND_FDIM

    n_atoms = 1  
    n_bonds = 1  
    a_scope = [] 
    b_scope = []

    f_atoms = [[0] * atom_fdim]  
    f_bonds = [[0] * bond_fdim]  
    a2b = [[]]  
    b2a = [0]  
    b2revb = [0]
    bonds = [[0,0]]

    for mol_graph in mol_graphs:
        f_atoms.extend(mol_graph['f_atoms'])
        f_bonds.extend(mol_graph['f_bonds'])

        for a in range(len(mol_graph['f_atoms'])):
            a2b.append([b + n_bonds for b in mol_graph['a2b'][a]]) #  if b!=-1 else 0

        for b in range(len(mol_graph['f_bonds'])):
            b2a.append(n_atoms + mol_graph['b2a'][b])
            b2revb.append(n_bonds + mol_graph['b2revb'][b])
            bonds.append([b2a[-1], 
                            n_atoms + mol_graph['b2a'][mol_graph['b2revb'][b]]])
        a_scope.append((n_atoms, len(mol_graph['f_atoms'])))
        b_scope.append((n_bonds, len(mol_graph['f_bonds'])))
        n_atoms += len(mol_graph['f_atoms'])
        n_bonds += len(mol_graph['f_bonds'])
        
    bonds = np.array(bonds).transpose(1,0)
        
    max_num_bonds = max(1, max(len(in_bonds) for in_bonds in a2b)) # max with 1 to fix a crash in rare case of all single-heavy-atom mols
        
    f_atoms = torch.FloatTensor(f_atoms)
    f_bonds = torch.FloatTensor(f_bonds)
    a2b = torch.LongTensor([a2b[a][:max_num_bonds] + [0] * (max_num_bonds - len(a2b[a])) for a in range(n_atoms)])
    b2a = torch.LongTensor(b2a)
    bonds = torch.LongTensor(bonds)
    b2revb = torch.LongTensor(b2revb)

    return (f_atoms, f_bonds, a2b, b2a, b2revb, a_scope) 

def read_pair_len(path):
    with open(f"{path}/mol_graph_0.pkl", 'rb') as handle:
        molgraph = pickle.load(handle)
    pocket_len, ligand_len, pair_len = molgraph['l_l_x'].shape
    return pocket_len, ligand_len, pair_len