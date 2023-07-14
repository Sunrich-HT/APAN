import os

from pathlib import Path

import numpy as np
import pandas as pd
import deepchem as dc
import yaml

from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter
# from atom_pair_typing import do_pair_encoding
from new_atom_pair_typing import do_pair_encoding


if __name__ == '__main__':

    parser = ArgumentParser(
        description="Atom Pair encoding for ligand-based model",
        formatter_class=ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument('--task', type=str, default='bbbp',
#                         choices=['esol', 'freesolv', 'lipo', 'bbbp', 'clintox', 'sider', 'tox21', 
#                         'qm7', 'qm9'],
                        help='dataset type.')
    parser.add_argument('--dataset-type', type=str, default='classification',
                        choices=['classification', 'regression'],
                        help='dataset type.')            
    parser.add_argument('--data-index', type=Path, default='data_index/',
    help='Path of the divided data')
    parser.add_argument('--saved-dir', type=Path, default='data/',
    help='The path to save the data after atom-pair-encoding')
    
    parser.add_argument('--separate_train_path', type=str,
                        help='Path to separate train set, optional')
    parser.add_argument('--separate_valid_path', type=str,
                        help='Path to separate valid set, optional')                    
    parser.add_argument('--separate_test_path', type=str,
                        help='Path to separate test set, optional')
    
    args, unknown = parser.parse_known_args()
    
    dataset_type = args.dataset_type
    task = args.task
    saved_dir = args.saved_dir
    
    data_index = args.data_index
    for fold in range(1):
        train_path = f'{data_index}/{task}/fold_{fold}_train.csv'
        valid_path = f'{data_index}/{task}/fold_{fold}_valid.csv'
        test_path = f'{data_index}/{task}/fold_{fold}_test.csv'
        path = [train_path, valid_path, test_path]
        for filename, set_name in zip([train_path, valid_path, test_path], [f"{saved_dir}/{task}/train_data_fold_{fold}", f"{saved_dir}/{task}/valid_data_fold_{fold}", f"{saved_dir}/{task}/test_data_fold_{fold}"]):
            do_pair_encoding(args, path, fold, filename, set_name)
