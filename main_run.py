# z cycle
import scanpy as sc
import pandas as pd
from utils_design import set_seed, load_config
from trainer_evaluater import run_unseen_celltype
import wandb
import torch
import warnings
warnings.filterwarnings("ignore")
# import os
import gc
import json
from typing import Dict, Any




cell_to_pred = 'CD4T'
params = load_config(cell_to_pred=cell_to_pred)
set_seed(params['seed'])
adata = sc.read(params['adata_path'])
emdata = sc.read(params['emdata_path'])
df,pred,ctrla,stima,test_za = run_unseen_celltype(adata, emdata, cell_to_pred, params, params['dataname'])


