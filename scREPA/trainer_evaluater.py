import scanpy as sc
import torch
import gc
import pandas as pd
from dataset_design import AnnDataSet
from model import scDistillOTC
from utils_design import normalize_embedding
import evaluate



def evaluate_model(data_name, eval_adata, cell_to_pred, key_dic, random_seed=0):
    return evaluate.evaluate(data_name=data_name,
                             eval_adata=eval_adata,
                             key_dic=key_dic,
                             random_seed=random_seed,
                             )
    # return evaluate.evaluate_adata(
    #                             eval_adata=eval_adata,
    #                             cell_type=cell_to_pred,
    #                             key_dic=key_dic,
    #                             random_seed=random_seed
    #                         )


def run_unseen_celltype(adata, emdata, cell_to_pred, params, dataname='PBMC',fm='X_scGPT',key_dic=None):
    key_dic = params['key_dic']
    em = normalize_embedding(emdata.obsm[fm])
    adata.obsm['fm'] = em
    adata.obs_names_make_unique()
    # Sensitivity analysis
    if params['sub'] is not None:
        sampled_indices = (
            adata.obs
            .groupby(['cell_type', 'condition'])
            .sample(frac=params['sub'], random_state=params['seed'])  # sampling 10% fix seed
            .index
        )
        adata_sampled = adata[sampled_indices, :]
        adata = adata_sampled
    
    model = scDistillOTC(input_dim=adata.n_vars,
                     latent_dim=params['latent_dim'],
                     hidden_dim=params['hidden_dim'],
                     noise_rate=params['noise_rate'],
                     num_heads=params['num_heads'],
                     cycle_weight=params['cycle_weight'],
                     KD_weight=params['KD_weight'],
                     device=params['device'])

    model = model.to(model.device)
    train = adata[~((adata.obs[key_dic['cell_type_key']] == cell_to_pred) &
                    (adata.obs[key_dic['condition_key']] == key_dic['stim_key']))].copy()
    
    model.train_scDistillOTC(train, epochs=params['epochs'], lr=params['lr'],
                      weight_decay=params['weight_decay'], batch_size=params['batch_size'], 
                      wandb_run=params.get('wandb_run', None))

    pred,ctrla,stima,test_za = model.predict_new(train_adata=train, cell_to_pred=cell_to_pred,
                                      key_dic=key_dic, ratio=0.005)
    
    gt = adata[(adata.obs[key_dic['cell_type_key']] == cell_to_pred)]
    eval_adata = gt.concatenate(pred)
    
    df = evaluate_model(dataname, eval_adata, cell_to_pred, key_dic)
    del model
    return df,pred,ctrla,stima,test_za
