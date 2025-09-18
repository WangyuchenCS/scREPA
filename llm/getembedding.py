from pathlib import Path
import warnings
# ignore warnings
warnings.filterwarnings("ignore")
import scanpy as sc
import scib
import numpy as np
import sys
sys.path.insert(0, "../")
import scgpt as scg
import matplotlib.pyplot as plt
plt.style.context('default')

model_dir = Path("/home/grads/ywang2542/Perturbation/Embedding/scGPT_CP")

data_dir = '/home/grads/ywang2542/Perturbation/scPRAM/data/'
data_name = "train_zheng"
# eval_path =  + 'train_zheng.h5ad'

sample_data_path = data_dir + data_name + ".h5ad"


adata = sc.read_h5ad(sample_data_path)
adata.var['gene_symbol'] = adata.var.index.values
gene_col = "gene_symbol"
if "cell_label" in adata.obs.columns:
    cell_type_key = "cell_label"
else:
    cell_type_key = "cell_type"
batch_key = "condition"
N_HVG = 7000

org_adata = adata.copy()

# preprocess should confirm the data is normalized and log1p transformed
# sc.pp.normalize_total(adata, target_sum=1e4)
# sc.pp.log1p(adata)
# highly variable genes
sc.pp.highly_variable_genes(adata, n_top_genes=N_HVG, flavor='seurat_v3')
adata = adata[:, adata.var['highly_variable']]

embed_adata = scg.tasks.embed_data(
    adata,
    model_dir,
    gene_col=gene_col,
    batch_size=2,
)  
# save the embedding
embed_adata.write_h5ad(data_dir + data_name + "_scgpt_embed.h5ad")

