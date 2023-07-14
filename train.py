from argparse import ArgumentParser, Namespace
from email import utils

import os
import math

from pathlib import Path

import cupy as cp
import numpy as np

from tqdm import tqdm

import torch

import torch.nn.functional as F

from torch.utils.data import DataLoader
from torch.cuda.amp.grad_scaler import GradScaler
from torch.optim import Adam

import transformers

from transformers import AdamW, get_linear_schedule_with_warmup

from ignite.engine import Engine, Events
from ignite.metrics import RunningAverage
from ignite.handlers import ModelCheckpoint, global_step_from_engine, EarlyStopping
from ignite.contrib.handlers import ProgressBar

from sklearn.model_selection import KFold
from sklearn.metrics import roc_auc_score

from utils.optim import RAdam
from utils.dataset import ISFDataset
from utils.mol_graph_feature import ATOM_FDIM, BOND_FDIM, get_mol_graph_fea, read_pair_len
# from utils.atom_pair_typing import cur_node_num 
from utils.scaler import StandardScaler
# from utils.nn_utils import initialize_weights, build_lr_scheduler
from utils.nn_utils import initialize_weights
from models.isf import SFModule


try:
    from torch.cuda.amp import autocast
    autocast_available = True
except ImportError:
    class autocast:
        def __init__(self, enabled=True): pass
        def __enter__(self): return self
        def __exit__(self, exc_type, exc_value, exc_traceback): pass
    autocast_available = False

    

alpha = 0.8
direction = 'best-r-score'  # 'best-loss'  
if __name__ == '__main__':
    from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter
    import yaml
    import wandb

    parser = ArgumentParser(
        description="Trainer script",
        formatter_class=ArgumentDefaultsHelpFormatter,
    )
    
    parser.add_argument('--task', type=str, default='esol',
        help='esol, freesolv, ...')
    parser.add_argument('--config', type=Path, default='configs/config.yaml',
        help='Use the following config for hparams.')
    parser.add_argument('--data', type=Path, default='data/',
        help='data path')    
    parser.add_argument('--checkpoint', type=str,
        help='Warm-start from a previous fine-tuned checkpoint.')
    parser.add_argument('--device', type=str, default='cuda',
                        help="Device. 'cpu', 'cuda', 'cuda:<n>'.")
    parser.add_argument('--distributed', action='store_true',
        help='Distributed training with apex.')
    parser.add_argument('--mp', action='store_true',
        help='Mixed Precision (MP)')
    parser.add_argument('--fold', type=int, default=0,
        help='-1,0,1,2,3,4,5')  

    args, unknown = parser.parse_known_args()
    
    with args.config.open() as y:
        config = yaml.load(y, Loader=yaml.FullLoader)
    
    print(config)
    
    if args.distributed:
        args.mp = True
        parser.add_argument('--local_rank', default=0, type=int,
                    help='node rank for distributed training')
        args = parser.parse_args()
        torch.cuda.set_device(args.local_rank)

        torch.distributed.init_process_group(backend='nccl',
                                            init_method='env://')
        torch.backends.cudnn.benchmark = True

    FOLD = args.fold
    device = args.device
    train_Ys = np.load(f"./{args.data}/{args.task}/train_data_fold_{FOLD}-Y.npy")
    
    if config['Regression']:
#         train_target = train_Ys.tolist()
#         data_scaler = StandardScaler().fit(train_target)
        train_mean = np.mean(train_Ys)
        train_std = np.std(train_Ys)
    
    valid_Ys = np.load(f"./{args.data}/{args.task}/valid_data_fold_{FOLD}-Y.npy")
    # valid_target = valid_Ys.tolist()
    test_Ys = np.load(f"./{args.data}/{args.task}/test_data_fold_{FOLD}-Y.npy")
    # test_target = test_Ys.tolist()
    # all_target = train_target + valid_target + test_target
    # data_scaler = StandardScaler().fit(all_target)
    
    pocket_len, ligand_len, pair_len = read_pair_len(f"./{args.data}/{args.task}/train_data_fold_{FOLD}")
    
    output_size = train_Ys.shape[-1]

    train_dataset = ISFDataset(f"./{args.data}/{args.task}/train_data_fold_{FOLD}", train_Ys)
    valid_dataset = ISFDataset(f"./{args.data}/{args.task}/valid_data_fold_{FOLD}", valid_Ys)
    test_dataset = ISFDataset(f"./{args.data}/{args.task}/test_data_fold_{FOLD}", test_Ys)
    
    print(len(train_dataset), len(valid_dataset), len(test_dataset))
    
    task = args.task
    
    if config['log_wandb']:
        import wandb
        wandb.init(project="scoring_function", entity="haotong")
    
    preds = []
    trues = []
    model = SFModule(config['msa_blocks_num'], config['evo_blocks_num'], output_size, pocket_len, ligand_len, config['embed_dims'], config['hidden_dims'], config['attention_heads'], pair_len, ATOM_FDIM, ATOM_FDIM + BOND_FDIM)
    # initialize_weights(model)
    model.to(device)

    checkpoint = torch.load('pretrained_model.pt', map_location=torch.device('cpu'))

    pretrained_dict = checkpoint['model']
    model_dict = model.state_dict()

    # 1. filter out unnecessary key
    pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict}

    # 2. overwrite entries in the existing state dict
    model_dict.update(pretrained_dict) 

    # # 3. load the new state dict
    model.load_state_dict(model_dict)  

    optimizer = RAdam(model.parameters(), lr=config['learning_rate'], weight_decay=config['weight_decay'])
    
    if args.checkpoint:
        checkpoint = args.checkpoint
    else:
        checkpoint = None


    if checkpoint is not None:
        optimizer.load_state_dict(torch.load(checkpoint)['optimizer'])
    
#     optimizer = AdamW(model.parameters(), lr=config['learning_rate'])
        
    scheduler = transformers.get_cosine_schedule_with_warmup(
        optimizer,
        num_warmup_steps=config['warmup_steps'],
        num_training_steps=config['training_steps'])
    
#     scheduler = transformers.get_constant_schedule_with_warmup(
#             optimizer,
#             num_warmup_steps=config['warmup_steps'])
    
    if args.mp:
        from apex import amp
        model, optimizer = amp.initialize(model, optimizer, opt_level="O1")

    if args.distributed:
        from apex.parallel import DistributedDataParallel
        model = DistributedDataParallel(model)

    scaler = GradScaler(enabled=False)

    def train_step(engine, batch):
        model.train()
        optimizer.zero_grad()
        # x, smi, y = batch
        x, y = batch
        if config['Regression']: 
#             train_y = data_scaler.transform(y.tolist()).tolist()
            y = torch.Tensor([[0 if x is None else x for x in tb] for tb in y])
        
        else:
            y = torch.Tensor([[0 if np.isnan(x) else x for x in tb] for tb in y.tolist()])
            y = y.to(device)
#             mask = torch.Tensor([[x is not None for x in tb] for tb in y.tolist()])、
            
            mask = torch.Tensor([[not np.isnan(x) for x in tb] for tb in y.tolist()])
            class_weights = torch.ones(y.shape)
            mask = mask.to(device)
            class_weights = class_weights.to(device)
        
        p_l_x, l_l_x, d_ij, index = x[0].float().to(device), x[1].float().to(device), x[2].float().to(device), x[3].float().to(device)
        
        if not config['use_distance']:
            d_ij = None
        mol_fea = get_mol_graph_fea(index, train_dataset.x_path)
        y = y.float().to(device)


        with autocast(enabled=False):
            pred_y = model(p_l_x, l_l_x, d_ij, mol_fea)
            
            if config['Regression']:
                y, pred_y = (y.flatten() - train_mean) / train_std, pred_y.flatten()
            
            if config['Regression']:
                if args.task in ['qm7', 'qm8', 'qm9']:
                    criterion = torch.nn.L1Loss()
                    loss = criterion(pred_y, y)
                else:
                    criterion = torch.nn.MSELoss()
                    loss = torch.sqrt(criterion(pred_y, y))
            else:
                criterion = torch.nn.BCEWithLogitsLoss(reduction='none')
                loss = criterion(pred_y, y) * class_weights * mask
                loss = loss.sum() / mask.sum()
        
        if args.mp:
            with amp.scale_loss(loss, optimizer) as scaled_loss:
                scaler.scale((scaled_loss / config['accum_steps'])).backward()
        else:
            scaler.scale((loss / config['accum_steps'])).backward()
        return loss.item()



    @torch.no_grad()
    def eval_step(engine, batch):
        model.eval()
        x, y = batch
        p_l_x, l_l_x, d_ij, index = x[0].float().to(device), x[1].float().to(device), x[2].float().to(device), x[3].float().to(device)
        
        if not config['use_distance']:
            d_ij = None
        
        y = y.float().to(device)
        mol_fea = get_mol_graph_fea(index, valid_dataset.x_path)
        
        if config['Regression']:
#             cur_y = torch.squeeze(y,dim=-1)
#             batch_pred_y = model(p_l_x, l_l_x, d_ij, mol_fea)
#             batch_pred_y_cur = data_scaler.inverse_transform(batch_pred_y.data.cpu().numpy())
#             batch_pred_y = np.squeeze(batch_pred_y, axis=-1).tolist()
#             pred_y = torch.Tensor([[0 if x is None else x for x in tb] for tb in batch_pred_y_cur.tolist()]).to(device)
            
            y_pred = model(p_l_x, l_l_x, d_ij, mol_fea)
            y_true = y.cpu().numpy().reshape(-1)  # (batch, task_numbers)
            y_pred = y_pred.detach().cpu().numpy().reshape(-1)  # (batch, task_numbers)
            
            preds.extend((y_pred * train_std + train_mean).tolist())
            trues.extend(y_true.tolist())
                   
            if args.task in ['qm7', 'qm8', 'qm9']:
                criterion = torch.nn.L1Loss()
                y_true, y_pred = y_true.flatten(), (y_pred.flatten() * train_std + train_mean).tolist()
#                 loss = criterion(y_pred, y_true)
                loss = F.l1_loss(torch.tensor(y_pred), torch.tensor(y_true), reduction='mean')
            else:
                criterion = torch.nn.MSELoss()
                y_true, y_pred = y_true.flatten(), (y_pred.flatten() * train_std + train_mean).tolist()
#                 loss = criterion(pred_y, y)
                loss = np.sqrt(F.mse_loss(torch.tensor(y_pred), torch.tensor(y_true), reduction='mean'))
#                 loss = torch.sqrt(criterion(pred_y, y))
        else:
        
            y = torch.Tensor([[0 if np.isnan(x) else x for x in tb] for tb in y.tolist()])
            y = y.to(device)
            
#             mask = torch.Tensor([[x is not None for x in tb] for tb in y.tolist()])
            mask = torch.Tensor([[not np.isnan(x) for x in tb] for tb in y.tolist()])
    
            class_weights = torch.ones(y.shape)
            mask = mask.to(device)
            class_weights = class_weights.to(device)
            pred_y = model(p_l_x, l_l_x, d_ij, mol_fea)
            
            criterion = torch.nn.BCEWithLogitsLoss(reduction='none')
            loss = criterion(pred_y, y) * class_weights * mask
            loss = loss.sum() / mask.sum()
            preds.extend(pred_y.tolist())
            trues.extend(y.tolist())
        
        return loss.item()


    trainer = Engine(train_step)
    evaluator = Engine(eval_step)


    @trainer.on(Events.STARTED)
    def update(engine):
        print('training started!')


    @trainer.on(Events.EPOCH_COMPLETED)
    @trainer.on(Events.ITERATION_COMPLETED(every=config['accum_steps']))
    
    def update(engine):
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), config['grad_norm'])
        scaler.step(optimizer)
        scaler.update()
        optimizer.zero_grad()
        scheduler.step()
    
    
    @trainer.on(Events.EPOCH_COMPLETED)
    def log_trn_loss(engine):
        log_msg = f"training epoch: {engine.state.epoch}"
        if config['Regression']:
            if args.task in ['qm7', 'qm8', 'qm9']:
                log_msg += f" | mae_loss: {engine.state.metrics['trn_rmse_loss']:.3f}"
            else:
                log_msg += f" | rmse_loss: {engine.state.metrics['trn_rmse_loss']:.3f}"
        else:
            log_msg += f" | loss: {engine.state.metrics['trn_loss']:.3f}"
        print(log_msg)
    
    @trainer.on(Events.EPOCH_COMPLETED)
    def run_dev_eval(engine):
        valid_dataloader.device = next(model.parameters()).device
        evaluator.run(valid_dataloader)
    
    if config['Regression']:
        if not config['best_loss']: 
            @evaluator.on(Events.EPOCH_COMPLETED)
            
            def R_score_eval(engine):
                R_score = np.corrcoef(trues, preds)[0, 1]
                engine.state.metrics['dev_r_score'] = R_score
                engine.state.metrics['dev_rmse+r_score'] = alpha * (1 - R_score) + (1 - alpha) * engine.state.metrics['dev_rmse_loss']
                preds.clear()
                trues.clear()
        else:
            @evaluator.on(Events.EPOCH_COMPLETED)
            def R_score_eval(engine):
                preds.clear()
                trues.clear()
    else:
        if not config['best_loss']:    
            @evaluator.on(Events.EPOCH_COMPLETED)
            def R_score_eval(engine):
                # Filter out empty targets
                # valid_preds and valid_targets have shape (num_tasks, data_size)
                valid_preds = [[] for _ in range(output_size)]
                valid_targets = [[] for _ in range(output_size)]
                for i in range(output_size):
                    for j in range(len(preds)):
                        if trues[j][i] is not None:  # Skip those without targets
                            valid_preds[i].append(preds[j][i])
                            valid_targets[i].append(trues[j][i])
                # Compute metric
                results = []
                for i in range(output_size):
                    # # Skip if all targets or preds are identical, otherwise we'll crash during classification
                    nan = False
                    if all(target == 0 for target in valid_targets[i]) or all(target == 1 for target in valid_targets[i]):
                        nan = True
                        # info('Warning: Found a task with targets all 0s or all 1s')
                    if all(pred == 0 for pred in valid_preds[i]) or all(pred == 1 for pred in valid_preds[i]):
                        nan = True
                        # info('Warning: Found a task with predictions all 0s or all 1s')

                    if nan:
                        results.append(float('nan'))
                        continue

                    if len(valid_targets[i]) == 0:
                        continue
                    results.append(roc_auc_score(valid_targets[i], valid_preds[i]))
                avg_val_score = np.nanmean(results)
                engine.state.metrics['dev_auc'] = avg_val_score
                engine.state.metrics['dev_loss+auc'] = alpha * (1 - avg_val_score) + (1 - alpha) * engine.state.metrics['dev_loss']
                preds.clear()
                trues.clear()
        else:
            @evaluator.on(Events.EPOCH_COMPLETED)
            def R_score_eval(engine):
                preds.clear()
                trues.clear()
    
    @evaluator.on(Events.EPOCH_COMPLETED)
    def log_dev_loss(engine):
        log_msg = f"dev epoch: {trainer.state.epoch}"
        if config['Regression']:
            if args.task in ['qm7', 'qm8', 'qm9']:
                log_msg += f" | mae_loss: {engine.state.metrics['dev_rmse_loss']:.3f}"
            else:
                log_msg += f" | rmse_loss: {engine.state.metrics['dev_rmse_loss']:.3f}"
            if not config['best_loss']:
                if args.task in ['qm7', 'qm8', 'qm9']:
                    log_msg += f" | r_score: {engine.state.metrics['dev_r_score']:.3f}"
                    log_msg += f" | mae+r_score: {engine.state.metrics['dev_rmse+r_score']:.3f}"
                else:
                    log_msg += f" | r_score: {engine.state.metrics['dev_r_score']:.3f}"
                    log_msg += f" | rmse+r_score: {engine.state.metrics['dev_rmse+r_score']:.3f}"
                preds.clear()
                trues.clear()
        else:
            log_msg += f" | loss: {engine.state.metrics['dev_loss']:.3f}"
            if not config['best_loss']:
                log_msg += f" | auc: {engine.state.metrics['dev_auc']:.3f}"
                log_msg += f" | auc+r_score: {engine.state.metrics['dev_loss+auc']:.3f}"
                preds.clear()
                trues.clear()
        print(log_msg)
    
    @trainer.on(Events.EPOCH_COMPLETED(every=1))
    def run_test(engine):
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
#                     cur_target= torch.squeeze(target,dim=-1)
#                     batch_pred_y = model(p_l_x, l_l_x, d_ij, mol_fea)
                   
                    y_pred = model(p_l_x, l_l_x, d_ij, mol_fea)
                    y_true = target.cpu().numpy().reshape(-1)  # (batch, task_numbers)
                    y_pred = y_pred.detach().cpu().numpy().reshape(-1)  # (batch, task_numbers)
            
                    preds.extend((y_pred * train_std + train_mean).tolist())
                    trues.extend(y_true.tolist())
#                     batch_pred_y_cur = data_scaler.inverse_transform(batch_pred_y.data.cpu().numpy())
#                     batch_pred_y = np.squeeze(batch_pred_y, axis=-1).tolist()
#                     pred_y = torch.Tensor([[0 if x is None else x for x in tb] for tb in batch_pred_y_cur.tolist()]).to(device)
                    # preds.append(float(pred_y.data.detach().cpu().numpy()))
                    # trues.append(float(target.detach().cpu().numpy()))
                    
            
#                     preds.extend(batch_pred_y)
#                     trues.extend(cur_target.data.cpu().numpy().tolist())
                    if args.task in ['qm7', 'qm8', 'qm9']:
                        y_true, y_pred = y_true.flatten(), (y_pred.flatten() * train_std + train_mean).tolist()
#                 loss = criterion(pred_y, y)
                        loss = F.l1_loss(torch.tensor(y_pred), torch.tensor(y_true), reduction='mean')
                    else:
#                         criterion = torch.nn.MSELoss()
#                         loss = criterion(pred_y, target)
                        y_true, y_pred = y_true.flatten(), (y_pred.flatten() * train_std + train_mean).tolist()
#                 loss = criterion(pred_y, y)
                        loss = np.sqrt(F.mse_loss(torch.tensor(y_pred), torch.tensor(y_true), reduction='mean'))
                    test_running_loss += loss.item()
                    # print("ture label:{}; pred label:{}".format(target, pred_y.data))
                test_loss = test_running_loss / counter
                if args.task in ['qm7', 'qm8', 'qm9']:
                    print(f"Test MAE is {test_loss:.4f}")
                else:
                    print(f"Test RMSE is {math.sqrt(test_loss):.3f}")
                print(f"Test R-Value is {np.corrcoef(trues, preds)[0, 1]}")
                engine.state.metrics['test_rmse_loss'] = math.sqrt(test_loss)
                engine.state.metrics['test_r_score'] = np.corrcoef(trues, preds)[0, 1]
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
#                     target = target.float().to(device)

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

    pbar = ProgressBar()
    pbar.attach(trainer)
    pbar.attach(evaluator)
    
    if config['Regression']:
        RunningAverage(output_transform=lambda out: out).attach(trainer, 'trn_rmse_loss')
        RunningAverage(output_transform=lambda out: out).attach(evaluator, 'dev_rmse_loss')
        
        if config['best_loss']:
            prefix = 'best-loss-rmse'
            score_function = lambda x: -evaluator.state.metrics['dev_rmse_loss']
        else:
            # prefix = 'best-r-score'
            prefix = 'best-rmse+r-score'
            score_function = lambda x: -evaluator.state.metrics['dev_rmse+r_score']
    else:
        RunningAverage(output_transform=lambda out: out).attach(trainer, 'trn_loss')
        RunningAverage(output_transform=lambda out: out).attach(evaluator, 'dev_loss')
        if config['best_loss']:
            prefix = 'best-loss'
            score_function = lambda x: -evaluator.state.metrics['dev_loss']
        else:
            prefix = 'best-auc'
            score_function = lambda x: -evaluator.state.metrics['dev_loss+auc']
    
    if config['log_wandb']:  
        from ignite.contrib.handlers.wandb_logger import WandBLogger   
        wandb_logger = WandBLogger(
            project="pytorch-ignite-integration",
            name="pocket-ligand-bind",
            config={"max_epochs": 100, "batch_size": 4},
            tags=["pytorch-ignite", "pocket-ligand-bind"]
        )
        wandb_logger.attach_output_handler(
            trainer,
            event_name=Events.ITERATION_COMPLETED,
            tag="iterations/trn_loss",
            output_transform=lambda loss: loss
        )
        if config['Regression']:
            metric_names_trn = ['trn_rmse_loss']
            metric_names_dev = ['dev_rmse_loss']
            if not config['best_loss']:
                metric_names_dev.append('dev_r_score')
        else:
            metric_names_trn = ['trn_loss']
            metric_names_dev = ['dev_loss']
            if not config['best_loss']:
                metric_names_dev.append('dev_auc')
        
        wandb_logger.attach_output_handler(
            trainer,
            event_name=Events.EPOCH_COMPLETED,
            tag="epochs",
            metric_names=metric_names_trn,
            global_step_transform=lambda *_: trainer.state.iteration,
        )

        wandb_logger.attach_output_handler(
            evaluator,
            event_name=Events.EPOCH_COMPLETED,
            tag="epochs",
            metric_names=metric_names_dev,
            global_step_transform=lambda *_: trainer.state.iteration,
        )
        @trainer.on(Events.ITERATION_COMPLETED)

        def wandb_log_lr(engine):
            wandb.log({'lr': scheduler.get_last_lr()[0]}, step=engine.state.iteration)
    
    
    to_save = {'model': model, 'optimizer': optimizer}
    root = f'output/{task}/'
    try:
        os.mkdir(root)
    except:
        pass
    where_checkpoints = root + str(len(list(Path(root).iterdir())))
    try:
        os.mkdir(where_checkpoints)
    except:
        pass

    handler = ModelCheckpoint(
                where_checkpoints,
                prefix,
                n_saved=1,
                create_dir=True,
                score_function=score_function,
                global_step_transform=global_step_from_engine(trainer),
                require_empty=False
            )
    handler_1 = EarlyStopping(patience=config['early_stopping_step'], score_function=score_function, trainer=trainer)
    evaluator.add_event_handler(Events.EPOCH_COMPLETED, handler, to_save)
    evaluator.add_event_handler(Events.COMPLETED, handler_1)
    device = next(model.parameters()).device

    if args.distributed:
        from torch.utils.data.distributed import DistributedSampler
        train_sampler = DistributedSampler(train_dataset)
        train_dataloader = DataLoader(train_dataset, batch_size=config['batch_size'], sampler=train_sampler)
    else:
        train_dataloader = DataLoader(train_dataset, batch_size=config['batch_size'], shuffle=True)
    
    valid_dataloader = DataLoader(valid_dataset, batch_size=config['batch_size'], shuffle=False)
    test_dataloader = DataLoader(test_dataset, batch_size=1, shuffle=False)
    
    train_dataloader.device = device
    valid_dataloader.device = device
    test_dataloader.device = device
    
    
    trainer.run(train_dataloader, max_epochs=config['max_epochs'])
