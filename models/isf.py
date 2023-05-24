from typing import List, Tuple, Union, Dict, Optional
from collections import OrderedDict
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from models.axial_attention import *
from models.evoformer import evoformer_base


class Node_Embeddings(nn.Module):
    def __init__(self, d_atom, d_emb, dropout):
        super(Node_Embeddings, self).__init__()
        self.lut = nn.Linear(d_atom, d_emb)
        self.dropout = nn.Dropout(dropout)
        self.d_emb = d_emb

    def forward(self, x):  # x.shape(batch, max_length, d_atom)
        return self.dropout(self.lut(x)) * math.sqrt(self.d_emb)

class Position_Encoding(nn.Module):
    def __init__(self, max_length, d_emb, dropout):
        super(Position_Encoding, self).__init__()
        self.dropout = nn.Dropout(dropout)
        self.pe = nn.Embedding(max_length, d_emb, padding_idx=0)

    def forward(self, x):
        return self.dropout(self.pe(x))  # (batch, max_length) -> (batch, max_length, d_emb)

class MSATransformerBlock(nn.Module):
    def __init__(self, rows, cols, embed_dims,
                 attention_heads, dim_linear_block=1024,
                 dropout=0.1, activation=nn.GELU):
        super().__init__()
        self.rows = rows
        self.columns = cols
        self.embed_dims = embed_dims
        self.attention_heads = attention_heads
        self.row_att = RowSelfAttention(self.embed_dims, self.attention_heads)
        self.col_att = ColumnSelfAttention(self.embed_dims, self.attention_heads)
        self.norm_1 = nn.LayerNorm(embed_dims)
        self.norm_2 = nn.LayerNorm(embed_dims)
        self.norm_3 = nn.LayerNorm(embed_dims)
        self.mlp = nn.Sequential(
            nn.LayerNorm(embed_dims),
            nn.Linear(embed_dims, dim_linear_block),
            activation(),
            nn.Dropout(dropout),
            nn.Linear(dim_linear_block, embed_dims),
            activation()
        )
    def forward(self, x):
        residual = x
        x, row_attn = self.row_att(self.norm_1(x))
        x += residual
        residual = x
        x, col_attn = self.col_att(self.norm_2(x))
        x += residual
        residual = x
        x = self.mlp(x)
        x += residual
        return x
def gaussian(x, mean, std):
    pi = 3.14159
    a = (2 * pi) ** 0.5
    return torch.exp(-0.5 * (((x - mean) / std) ** 2)) / (a * std)
def index_select_ND(source: torch.Tensor, index: torch.Tensor) -> torch.Tensor:
    """
    Selects the message features from source corresponding to the atom or bond indices in index.
    :param source: A tensor of shape (num_bonds, hidden_size) containing message features.
    :param index: A tensor of shape (num_atoms/num_bonds, max_num_bonds) containing the atom or bond
    indices to select from source.
    :return: A tensor of shape (num_atoms/num_bonds, max_num_bonds, hidden_size) containing the message
    features corresponding to the atoms/bonds specified in index.
    """
    index_size = index.size()  # (num_atoms/num_bonds, max_num_bonds)
    suffix_dim = source.size()[1:]  # (hidden_size,)
    final_size = index_size + suffix_dim  # (num_atoms/num_bonds, max_num_bonds, hidden_size)
    target = source.index_select(dim=0, index=index.view(-1))  # (num_atoms/num_bonds * max_num_bonds, hidden_size)
    target = target.view(final_size)  # (num_atoms/num_bonds, max_num_bonds, hidden_size)
    
    target[index==0] = 0
    return target
class BatchGRU(nn.Module):
    def __init__(self, hidden_size=300):
        super(BatchGRU, self).__init__()
        self.hidden_size = hidden_size
        self.gru  = nn.GRU(self.hidden_size, self.hidden_size, batch_first=True, 
                           bidirectional=True)
        self.bias = nn.Parameter(torch.Tensor(self.hidden_size))
        self.bias.data.uniform_(-1.0 / math.sqrt(self.hidden_size), 
                                1.0 / math.sqrt(self.hidden_size))
    def forward(self, node, a_scope):
        hidden = node
        message = F.relu(node + self.bias)
        MAX_atom_len = max([a_size for a_start, a_size in a_scope])
        # padding
        message_lst = []
        hidden_lst = []
        for i, (a_start, a_size) in enumerate(a_scope):
            if a_size == 0:
                assert 0
            cur_message = message.narrow(0, a_start, a_size)
            cur_hidden = hidden.narrow(0, a_start, a_size)
            hidden_lst.append(cur_hidden.max(0)[0].unsqueeze(0).unsqueeze(0))
            
            cur_message = torch.nn.ZeroPad2d((0,0,0,MAX_atom_len-cur_message.shape[0]))(cur_message)
            message_lst.append(cur_message.unsqueeze(0))
            
        message_lst = torch.cat(message_lst, 0)
        hidden_lst  = torch.cat(hidden_lst, 1)
        hidden_lst = hidden_lst.repeat(2,1,1)
        cur_message, cur_hidden = self.gru(message_lst, hidden_lst)
        
        # unpadding
        cur_message_unpadding = []
        for i, (a_start, a_size) in enumerate(a_scope):
            cur_message_unpadding.append(cur_message[i, :a_size].view(-1, 2*self.hidden_size))
        cur_message_unpadding = torch.cat(cur_message_unpadding, 0)
        
        message = torch.cat([torch.cat([message.narrow(0, 0, 1), message.narrow(0, 0, 1)], 1), 
                             cur_message_unpadding], 0)
        return message
class GaussianLayer(nn.Module):
    def __init__(self, K=128, edge_types=1024):
        super().__init__()
        self.K = K
        self.means = nn.Embedding(1, K)
        self.stds = nn.Embedding(1, K)
        self.mul = nn.Embedding(edge_types, 1)
        self.bias = nn.Embedding(edge_types, 1)
        nn.init.uniform_(self.means.weight, 0, 3)
        nn.init.uniform_(self.stds.weight, 0, 3)
        nn.init.constant_(self.bias.weight, 0)
        nn.init.constant_(self.mul.weight, 1)
    def forward(self, x):
#         x = x.expand(-1, -1, -1, self.K)
        mean = self.means.weight.float().view(-1)
        std = self.stds.weight.float().view(-1).abs() + 1e-5
        return gaussian(x.float(), mean, std).type_as(self.means.weight)
    
    
class NonLinearHead(nn.Module):
    """Head for simple classification tasks."""
    def __init__(
        self,
        input_dim,
        out_dim,
        hidden=None,
    ):
        super().__init__()
        hidden = input_dim if not hidden else hidden
        self.linear1 = nn.Linear(input_dim, hidden)
        self.linear2 = nn.Linear(hidden, out_dim)
        self.activation_fn = F.gelu
    def forward(self, x):
        x = self.linear1(x)
        x = self.activation_fn(x)
        x = self.linear2(x)
        return x
class SFModule(nn.Module):
    def __init__(self, num_blocks, num_evo_blocks, output_dim,
                 rows, cols, embed_dims, hidden_dims,
                 attention_heads, pair_len, atom_fdim, bond_fdim):
        super().__init__()
        self.num_blocks = num_blocks
        self.num_evo_blocks = num_evo_blocks
        self.output_dim = output_dim
        self.pair_len = pair_len
        self.atom_fdim = atom_fdim
        self.bond_fdim = bond_fdim
        self.rows = rows
        self.columns = cols
        self.embed_dims = embed_dims
        self.hidden_dims = hidden_dims
        self.attention_heads = attention_heads
        self.depth = 3
        self._modules: Dict[str, Optional['Module']] = OrderedDict()
        self.activation = F.gelu
        self.pair_embed = evoformer_base(self.pair_len, self.pair_len)
        self.input_linear = nn.Linear(self.pair_len, self.pair_len)
        self.gaussian_layer = GaussianLayer(self.pair_len)
        
        self.node_embed = Node_Embeddings(self.pair_len, self.pair_len, 0.0)
        self.pos_emb = Position_Encoding(25, self.pair_len, 0.0)

        #         self.gaussian_layer_1 = GaussianLayer(176)
        self.non_linear = NonLinearHead(self.pair_len, self.embed_dims)
        self.non_linear_1 = NonLinearHead(176, self.embed_dims, hidden=100)
        
        self.non_linear_x = NonLinearHead(16, self.hidden_dims, hidden=100)
        
        self.blocks = nn.ModuleList([MSATransformerBlock(self.rows, self.columns, self.embed_dims,
                                                         self.attention_heads) for _ in range(self.num_blocks)])
        
        self.evo_blocks = nn.ModuleList([evoformer_base(self.pair_len, self.pair_len) for _ in range(self.num_evo_blocks)])
        
        self.gru = nn.GRU(input_size=self.hidden_dims, hidden_size=self.hidden_dims,
             num_layers=1,  
             batch_first=True,  
             bidirectional=False,
             )
        self.W_i_atom = nn.Linear(self.atom_fdim, self.hidden_dims, bias=False)
        self.W_i_bond = nn.Linear(self.bond_fdim, self.hidden_dims, bias=False)
        self.W_o = nn.Linear(self.hidden_dims * 2, self.hidden_dims)
        self.act_func = nn.ReLU()
        
        # Dropout
        self.dropout_layer = nn.Dropout(p=0.0)
        self.gru_fea = BatchGRU(self.hidden_dims)
        
        self.lr = nn.Linear(self.hidden_dims * 3, self.hidden_dims, bias=False)
        for depth in range(self.depth-1):
            self._modules[f'W_h_{depth}'] = nn.Linear(self.hidden_dims, self.hidden_dims, bias=False)
        
        self.ffc1 = nn.Linear(self.embed_dims, 8)
        self.ffc2 = nn.Linear(8, 1)
        
        self.ffn = [self.dropout_layer, nn.Linear(self.hidden_dims, self.hidden_dims)]
        self.ffn.extend([self.act_func, self.dropout_layer, nn.Linear(self.hidden_dims, self.output_dim)])
        self.ffn = nn.Sequential(*self.ffn)
    
    def get_node_bond_fea(self, mol_fea):
        (f_atoms, f_bonds, a2b, b2a, b2revb, a_scope) = mol_fea
        f_atoms, f_bonds, a2b, b2a, b2revb = (
                    f_atoms.cuda(), f_bonds.cuda(), 
                    a2b.cuda(), b2a.cuda(), b2revb.cuda())
        input_atom = self.W_i_atom(f_atoms)  # num_atoms x hidden_size
        input_atom = self.act_func(input_atom)
        message_atom = input_atom.clone()
        
        input_bond = self.W_i_bond(f_bonds)  # num_bonds x hidden_size
        message_bond = self.act_func(input_bond)
        input_bond = self.act_func(input_bond)
        # Message passing
        for depth in range(self.depth - 1):
            agg_message = index_select_ND(message_bond, a2b)
            agg_message = agg_message.sum(dim=1) * agg_message.max(dim=1)[0]
            message_atom = message_atom + agg_message
            
            # directed graph
            rev_message = message_bond[b2revb]  # num_bonds x hidden
            message_bond = message_atom[b2a] - rev_message  # num_bonds x hidden
            
            message_bond = self._modules[f'W_h_{depth}'](message_bond)
            message_bond = self.dropout_layer(self.act_func(input_bond + message_bond))
        
        agg_message = index_select_ND(message_bond, a2b)
        agg_message = agg_message.sum(dim=1) * agg_message.max(dim=1)[0]
        agg_message = self.lr(torch.cat([agg_message, message_atom, input_atom], 1))
        agg_message = self.gru_fea(agg_message, a_scope)
        
        atom_hiddens = self.act_func(self.W_o(agg_message))  # num_atoms x hidden
        atom_hiddens = self.dropout_layer(atom_hiddens)  # num_atoms x hidden
        
        # Readout
        mol_vecs = []
        for i, (a_start, a_size) in enumerate(a_scope):
            if a_size == 0:
                assert 0
            cur_hiddens = atom_hiddens.narrow(0, a_start, a_size)
            mol_vecs.append(cur_hiddens.mean(0))
        mol_vecs = torch.stack(mol_vecs, dim=0)
        
        return mol_vecs  # B x H
        
    
    def forward(self, p_l_x, l_l_x, d_ij, mol_fea):
        
        # embedding层
        for block_idx, block in enumerate(self.evo_blocks):
            p_l_x_o, l_l_x_o = block(p_l_x, l_l_x)
        if d_ij != None:
            x = torch.mul(p_l_x_o, d_ij)
        else:
            x = p_l_x_o
        x = self.input_linear(self.activation(x))
        x = self.gaussian_layer(x)
        x = self.non_linear(x)  # [B,M,N,d]
        
        x = x.permute(1, 2, 0, 3)
        for block_idx, block in enumerate(self.blocks):
            x = block(x)
        # 卷积
        # x = x.permute(2, 3, 1, 0)   # [B,d,N,M]
        # x = self.conv_blocks(x)
        # mean
        x = x.permute(2, 0, 1, 3)
        x = x.mean(1)
        
        fea = self.get_node_bond_fea(mol_fea)
        x = self.non_linear_x(x)
        fea = torch.unsqueeze(fea, dim=0)
        x, hx = self.gru(x, fea)
        
        x = x.mean(1)
        
        x = self.ffn(x)
        
#         x = torch.sigmoid(x)
#         print(x.shape)
        return x
