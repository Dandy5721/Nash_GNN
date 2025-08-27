# Nash_GNN

A PyTorch / PyG implementation of Graph Neural Network models for **disease diagnosis on brain connectivity data** and **general graph-based analysis**.

- **Brain connectivity diagnosis**: models tailored for human connectome / disease datasets.
- **Graph-based analysis** (三类任务):
  1) **Node Classification** – standard full-batch transductive setting  
  2) **Graph Classification** – multiple graphs per dataset (e.g., TU benchmarks / subject-level connectomes)  
  3) **Large-scale Node Classification** – mini-batch / neighbor-sampling training for million-scale graphs

> Implementations include curvature-based models (Mean Curvature / Beltrami flow) and Nash-inspired GNN variants, all built on PyTorch Geometric.

---

## Tasks & Quick Start

### 1) Node Classification
```bash
python graph_node_classification.py \
  --dataset Cora --model beltrami --epochs 300 --lr 1e-3
