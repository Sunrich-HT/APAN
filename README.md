## Molecular property prediction

The main function of this repo is to implement Molecular Property Prediction(ligand-based model)

## Installation

```text
python=3.8
pytorch=1.12
rdkit
scikit-learn
xlrd
tqdm
pytorch-ignite
transformers==2.11.0
cupy-cuda113
deepchem
einops
pyyaml
wandb
apex
```
## Data preprocessing
```bash
python utils/data_fetch.py --dataset-type classification --task bbbp
python utils/data_fetch.py --dataset-type regression --task esol 
```

## Train

```bash
python train.py --config configs/config.yaml --task bbbp --fold 0 --mp
python train.py --config configs/config.yaml --task esol --fold 0 --mp
##distributed
CUDA_VISIBLE_DEVICES=0,1 python -m torch.distributed.launch --nproc_per_node=2 train.py --config configs/config.yaml --task esol --fold 0 --mp --distributed
```

## Test

```bash
python predict.py --config configs/config.yaml --task esol --device cuda --checkpoint ***.pt --fold 0 
##distributed
python predict.py --config configs/config.yaml --task esol --device cuda --checkpoint ***.pt --fold 0 --distributed
```
