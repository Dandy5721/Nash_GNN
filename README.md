# Nash_GNN

A PyTorch / PyG implementation of *Nash_GNN* for **disease diagnosis on brain connectivity data** and **general graph-based analysis**.

- **Brain connectivity diagnosis**: models tailored for human connectome / disease datasets.
- **Graph-based analysis**:
  1) **Node Classification** – standard full-batch transductive setting  
  2) **Graph Classification** – multiple graphs per dataset (e.g., TU benchmarks / subject-level connectomes)  
  3) **Large-scale Node Classification** – full-batch or mini-batch / neighbor-sampling training for million-scale graphs

> Implementations include curvature-based models (Mean Curvature / Beltrami flow) and Nash-inspired GNN variants, all built on PyTorch Geometric.

---

## Tasks & Quick Start

### 1) Node Classification
```bash
python graph_node_classification.py --grb_mode full --runs 1 --model GCN --time 3  --method euler --function icnn --gpu 1 --hidden_dim 128 --eval_robo_blk --step_size 1 --input_dropout 0.4 --batch_norm --add_source --grb_split --dataset Cora
```
### 2) Large-scale Node Classification
```bash
python -u graph_node_class_Nash_GNN.py --grb_mode full --runs 1 --model graphcon --time 3  --method euler --function icnn --gpu 1 --hidden_dim 128 --eval_robo_blk --step_size 1 --input_dropout 0.4 --batch_norm --add_source --grb_split --dataset arxiv
```

### 3) Graph Classification
```bash
python -u graph_classification.py -data mutag -batch 128 -hid_dim 64 -lr 0.01 -model 'GCN' -gpu 0 
```

### 4) Brain Connectomes Classification
```bash
# python -u graph_classification_human_connectomes.py --dataset 'adni' --runs 5 --model graphbel --gpu 0 
```
