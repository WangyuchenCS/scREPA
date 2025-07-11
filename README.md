# 🔬 scREPA: Predicting Single-Cell Perturbation Responses with Cycle-Consistent Representation Alignment

![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)

Official implementation of the paper:  
**“scREPA: Predicting Single-Cell Perturbation Responses with Cycle-Consistent Representation Alignment”**

---

## 🚀 Overview

scREPA is a generative deep learning framework designed to model and predict single-cell gene expression responses under perturbation. It addresses the challenge of limited and noisy scRNA-seq data by leveraging pretrained **single-cell foundation models (scFMs)** and introducing novel **representation alignment strategies**.

---

## 📋 Usage

A complete training and evaluation pipeline is provided in the `scREPA/` folder, including:

- 🧬 Loading and preprocessing scRNA-seq perturbation datasets  
- 🧠 Initializing and training the `scREPA` model  
- 🔮 Predicting perturbation responses for unseen cell types or conditions  
- 📊 Visualizing and evaluating model performance  

To run the full pipeline, simply execute:

```bash
python main_run.py
```
---

## 📋 Tutorial

run tutorial.ipynb

