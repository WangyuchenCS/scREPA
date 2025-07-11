import warnings
import numpy as np
import pandas as pd
import random
import torch
import torch.nn as nn
from torch import Tensor, optim
# import gdown
import scanpy as sc
import evaluate
# import models
import gc
import ot
from tqdm import tqdm
from tqdm.notebook import tqdm
from torch.distributions import Normal
from torch.distributions import kl_divergence as kl
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import normalize
from torch.optim.lr_scheduler import StepLR
# from dataset import AnnDataSet
from torch.utils.data import Dataset, DataLoader
warnings.filterwarnings('ignore')

import torch
from torch.utils.data import Dataset

class AnnDataSet(Dataset):
    def __init__(self, adata):
        self.data = adata.to_df().values
        self.embedding = adata.obsm['fm'] if 'fm' in adata.obsm else 0
        try:
            self.cell_type = adata.obs['cell_type']
        except KeyError:
            try:
                self.cell_type = adata.obs['cell_label']
            except KeyError:
                self.cell_type = adata.obs['louvain']
        unique_labels = self.cell_type.unique()
        self.label_mapping = {label: idx for idx, label in enumerate(unique_labels)}
        self.cell_type = self.cell_type.map(self.label_mapping).values

    def __getitem__(self, index):
        x = self.data[index, :]
        y = self.cell_type[index]
        e = self.embedding[index, :] if self.embedding is not None else 0
        return x, y, e

    def __len__(self):
        return self.data.shape[0]


