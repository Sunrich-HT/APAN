from operator import index
import pickle
import torch
import torch.nn as nn

from tqdm import tqdm
import numpy as np
import cupy as cp

from torch.utils.data import Dataset
from cupy._core.dlpack import toDlpack
from torch.utils.dlpack import from_dlpack
class ISFDataset(Dataset):
    
    def __init__(self, x_path, y, device='cuda'):
        super().__init__()
        
        self.x_path = x_path
        self.device = device
        self.y = y
        # self.args = args
        
    def __getitem__(self, index):
        with open(f"{self.x_path}/mol_graph_{index}.pkl", 'rb') as handle:
            molgraph = pickle.load(handle)
        
        T_ij = torch.from_numpy(molgraph['pair_one_coding']).to(self.device)
        l_l_x = torch.from_numpy(molgraph['l_l_x']).to(self.device)
        d_ij = torch.from_numpy(molgraph['d_ij']).to(self.device)
        
        del molgraph['pair_one_coding']
        del molgraph['l_l_x']
        del molgraph['d_ij']

        # if self.args.task in ['qm7', 'qm8', 'qm9']:
        #     d_ij = torch.from_numpy(molgraph['d_ij']).to(self.device)
        #     del molgraph['d_ij']
            
        return (T_ij, l_l_x, d_ij, index), self.y[index]
    
    def __len__(self):
        return self.y.shape[0]
