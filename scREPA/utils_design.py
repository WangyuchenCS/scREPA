import torch
import numpy as np
import random
import json
from typing import Dict, Any
import scanpy as sc

def set_seed(seed=0):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def normalize_embedding(emb):
    if not isinstance(emb, np.ndarray):
        emb = emb.toarray()
    mean = emb.mean(axis=0)
    std = emb.std(axis=0) + 1e-6
    return (emb - mean) / std


def load_config(cell_to_pred=None, config_file="/home/grads/ywang2542/Perturbation/Embedding/scREPA/config.json"):
    """
    Load configuration from JSON file 
    Args:
        cell_to_pred: Cell type
        config_file: Path to JSON configuration file
    
    Returns:
        Dictionary containing all configuration parameters
    """
    with open(config_file, 'r') as f:
        config = json.load(f)
    
    params = config["default"].copy()
    
    if cell_to_pred and cell_to_pred in config["cell_specific"]:
        params.update(config["cell_specific"][cell_to_pred])
    
    return params