# 🔬 scREPA: Predicting Single-Cell Perturbation Responses with Cycle-Consistent Representation Alignment

![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)

Official implementation of the paper:  
**“scREPA: Predicting Single-Cell Perturbation Responses with Cycle-Consistent Representation Alignment”**

> This repository contains code and pretrained resources for our paper:
> **"\[scOTM: A Deep Learning Framework for Predicting Single-Cell Perturbation Responses with Large Language Models]"**
> \[Yuchen Wang et al.]
---

## 🚀 Abstract

Modeling cellular perturbation responses is essential for understanding disease mechanisms and developing therapeutic strategies. Recent advances in single-cell foundation models (scFMs) offer a promising solution by providing biologically meaningful representations. Inspired by the success of REPresentation Alignment (REPA) in generative diffusion models, we propose scREPA, a novel framework for single-cell perturbation prediction that aligns the internal representations of a variational autoencoder (VAE)-based model with high-quality external representations from pretrained scFMs. Specifically, scREPA aligns VAE latent embeddings from noisy gene expression profiles with biologically meaningful embeddings from scFMs. We also propose Cycle-Consistent Representation Alignment by aligning the re-encoded embeddings of VAE-generated gene expression profiles with both original scFM representations and initial VAE embeddings, enforcing dual consistency and further improving representation quality. During inference, scREPA applies optimal transport to align the distributions of unpaired control and perturbed data, enabling robust prediction of cellular responses by minimizing mismatch.

---

## 📖 Overview

This project provides reproducible setup for results in the paper, including:

* Environment Setup
* Preprocessing and handling of single-cell RNA-seq data
* Extraction of LLM embeddings
* Model training and evaluation scripts

---

## 💼 Repository Structure

```
├── data/                   # Input datasets (.h5ad)
├── env/                    # Environment file
├── llm/                    # code for extracting embedding of LLM models
├── src/                    # Training, evaluation, utility scripts
├── tutorial.ipynb          # Full tutorial
└── README.md
```

---

## ⚙️ Environment Setup

This project is built with Conda and Python 3.8+. We recommend using the provided `.yml` file to fully replicate the environment.

### ✅ Install via Conda

```bash
conda env create -f scOTM.yml
conda activate scOTM
```

---

## 🌟 Extracting Embeddings from scGPT

This project requires **cell-level embeddings** extracted from the pretrained [scGPT](https://github.com/bowang-lab/scGPT) model.

### ✅ Step 1: Install scGPT

```bash
git clone https://github.com/bowang-lab/scGPT.git
cd scGPT
pip install -e .
```

### ⏬ Step 2: Download Pretrained Models

```bash
# Example: replace with correct IDs or links
wget --no-check-certificate 'https://drive.google.com/uc?export=download&id=1x1SfmFdI-zcocmqWAd7ZTC9CTEAVfKZq' -O best_model.pth
wget --no-check-certificate 'https://drive.google.com/uc?export=download&id=1jfT_T5n8WNbO9QZcLWObLdRG8lYFKH-Q' -O vocab.json
wget --no-check-certificate 'https://drive.google.com/uc?export=download&id=15TEZmd2cZCrHwgfE424fgQkGUZCXiYrR' -O args.json
```

### ⏬ Step 3: Run embedding extraction

The input must be an .h5ad file (AnnData) with the following:
Gene expression matrix (adata.X) should be log-normalized counts or CPM
Genes must be aligned to the vocabulary used by the pretrained model
Gene names must match vocab.json keys
Modify the input and output configuration as needed

```bash
python LLM/getembedding.py 

```


---
## 🔥 Model Training & Evaluation


A complete training and evaluation pipeline is provided in the `scREPA/` folder, including:

- 🧬 Loading and preprocessing scRNA-seq perturbation datasets  
- 🧠 Initializing and training the `scREPA` model  
- 🔮 Predicting perturbation responses for unseen cell types or conditions  
- 📊 Visualizing and evaluating model performance  

After preparing the embeddings, run model training:

```bash
python screpa/main_run.py
```


---

## 📋 Tutorial

run tutorial.ipynb

