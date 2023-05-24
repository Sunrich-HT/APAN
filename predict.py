import os
import math
import yaml
from pathlib import Path

import numpy as np
import pandas as pd

from tqdm import tqdm

import torch
from torch.utils.data import DataLoader

from models.isf import SFModule
from utils.nn_utils import initialize_weights
from utils.dataset import ISFDataset
from utils.mol_graph_feature import ATOM_FDIM, BOND_FDIM, get_mol_graph_fea, read_pair_len
from collections import OrderedDict
from utils.scaler import StandardScaler
from sklearn.metrics import roc_auc_score

if __name__ == '__main__':
    from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter
    
    parser = ArgumentParser(
        description='Script to predict Protein-Ligand Binding Affinity',
        formatter_class=ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument('--task', type=str, default='toxcast',
        help='esol, freesolv, ...')
    parser.add_argument('--config', type=Path, default='configs/config.yaml',
        help='Use the following config for hparams.')
    parser.add_argument('--data', type=Path, default='new_scaf_data/',
        help='data path')
    parser.add_argument('--checkpoint', type=str, default='output/toxcast/0/best-auc_checkpoint_66_0.7378.pt'
                        )
    parser.add_argument('--device', type=str, default='cuda',
                        help="Device. 'cpu', 'cuda', 'cuda:<n>'.")
    parser.add_argument('--pred-path', type=Path, default='test.csv')
    parser.add_argument('--distributed', action='store_true')
    parser.add_argument('--fold', type=int, default=0)
    
    args = parser.parse_args()
    with args.config.open() as y:
        config = yaml.load(y, Loader=yaml.FullLoader)
    
    FOLD = args.fold
    train_Ys = np.load(f"./{args.data}/{args.task}/train_data_fold_{FOLD}-Y.npy")
    if config['Regression']:
        train_target = train_Ys.tolist()
        data_scaler = StandardScaler().fit(train_target)
        
    test_Ys = np.load(f"./{args.data}/{args.task}/test_data_fold_{FOLD}-Y.npy")
    
    test_dataset = ISFDataset(f"./{args.data}/{args.task}/test_data_fold_{FOLD}", test_Ys)
    test_dataloader = DataLoader(test_dataset, batch_size=1, shuffle=False)
    
    pocket_len, ligand_len, pair_len = read_pair_len(f"./{args.data}/{args.task}/train_data_fold_{FOLD}")
    output_size = train_Ys.shape[-1]
    
    model = SFModule(config['msa_blocks_num'], config['evo_blocks_num'], output_size, pocket_len, ligand_len, config['embed_dims'], config['hidden_dims'], config['attention_heads'], pair_len, ATOM_FDIM, ATOM_FDIM + BOND_FDIM)
    
#     initialize_weights(model)
    checkpoint = torch.load(args.checkpoint, map_location=torch.device('cpu'))
    
    # pretrained_dict = checkpoint['model']
    # model_dict = model.state_dict()

    # # 1. filter out unnecessary key
    # pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict}

    # # 2. overwrite entries in the existing state dict
    # model_dict.update(pretrained_dict) 

    # # # 3. load the new state dict
    # model.load_state_dict(model_dict)  
    
    if args.distributed:
        new_state_dict = OrderedDict()
        for k, v in checkpoint['model'].items():
            name = k[7:]
            new_state_dict[name] = v
        model.load_state_dict(new_state_dict)
    else:
        model.load_state_dict(checkpoint['model'])
    
    # model.load_state_dict(checkpoint['model'])
    device = torch.device(args.device)
    model.to(device)
    if config['Regression']:
        counter = 0
        total = 0
        test_running_loss = 0.0
        preds = []
        trues = []
        prog_bar = tqdm(enumerate(test_dataloader), total=int(len(test_dataset) / test_dataloader.batch_size))
        with torch.no_grad():
            for i, (data, target) in prog_bar:
                counter += 1
                p_l_x, l_l_x, d_ij, index = data[0].float().to(device), data[1].float().to(device), data[2].float().to(device), data[3].float().to(device)
                if not config['use_distance']:
                    d_ij = None
                mol_fea = get_mol_graph_fea(index, test_dataset.x_path)
                target = target.float().to(device)
                cur_target= torch.squeeze(target,dim=-1)
                batch_pred_y = model(p_l_x, l_l_x, d_ij, mol_fea)

                batch_pred_y_cur = data_scaler.inverse_transform(batch_pred_y.data.cpu().numpy())
                batch_pred_y = np.squeeze(batch_pred_y, axis=-1).tolist()
                pred_y = torch.Tensor([[0 if x is None else x for x in tb] for tb in batch_pred_y_cur.tolist()]).to(device)
                # preds.append(float(pred_y.data.detach().cpu().numpy()))
                # trues.append(float(target.detach().cpu().numpy()))
                preds.extend(batch_pred_y)
                trues.extend(cur_target.data.cpu().numpy().tolist())
                if args.task in ['qm7', 'qm8', 'qm9']:
                    criterion = torch.nn.L1Loss()
                    loss = criterion(pred_y, target)
                else:
                    criterion = torch.nn.MSELoss()
                    loss = criterion(pred_y, target)
                test_running_loss += loss.item()
                # print("ture label:{}; pred label:{}".format(target, pred_y.data))
            test_loss = test_running_loss / counter
            if args.task in ['qm7', 'qm8', 'qm9']:
                print(f"Test MAE is {test_loss:.4f}")
            else:
                print(f"Test RMSE is {math.sqrt(test_loss):.3f}")
                print(f"Test R-Value is {np.corrcoef(trues, preds)[0, 1]}")

        df = pd.DataFrame(data={"preds": preds, "gts": trues})
        df.to_csv(args.pred_path)
    else:
        counter = 0
        total = 0
        test_running_loss = 0.0
        preds = []
        trues = []
        prog_bar = tqdm(enumerate(test_dataloader), total=int(len(test_dataset) / test_dataloader.batch_size))
        with torch.no_grad():
            for i, (data, target) in prog_bar:
                counter += 1
                p_l_x, l_l_x, d_ij, index = data[0].float().to(device), data[1].float().to(device), data[2].float().to(device), data[3].float().to(device)
                
                if not config['use_distance']:
                    d_ij = None

                target = torch.Tensor([[0 if np.isnan(x) else x for x in tb] for tb in target.tolist()])
                target = target.float().to(device)
                mol_fea = get_mol_graph_fea(index, test_dataset.x_path)
                pred_y = model(p_l_x, l_l_x, d_ij, mol_fea)

                preds.extend(pred_y.tolist())
                trues.extend(target.tolist())

            test_preds = [[] for _ in range(output_size)]
            test_targets = [[] for _ in range(output_size)]
            for i in range(output_size):
                for j in range(len(preds)):
                    if trues[j][i] is not None:  # Skip those without targets
                        test_preds[i].append(preds[j][i])
                        test_targets[i].append(trues[j][i])
            # Compute metric
            results = []
            for i in range(output_size):
                # # Skip if all targets or preds are identical, otherwise we'll crash during classification
                nan = False
                if all(target == 0 for target in test_targets[i]) or all(target == 1 for target in test_targets[i]):
                    nan = True
                    # info('Warning: Found a task with targets all 0s or all 1s')
                if all(pred == 0 for pred in test_preds[i]) or all(pred == 1 for pred in test_preds[i]):
                    nan = True
                    # info('Warning: Found a task with predictions all 0s or all 1s')

                if nan:
                    results.append(float('nan'))
                    continue

                if len(test_targets[i]) == 0:
                    continue
                results.append(roc_auc_score(test_targets[i], test_preds[i]))
            avg_val_score = np.nanmean(results)
            print(f"Test AUC is {avg_val_score:.3f}")
            preds = []
            trues = []
        df = pd.DataFrame(data={"preds": preds, "gts": trues})
        df.to_csv(args.pred_path)





