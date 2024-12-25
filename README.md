# TiGer
Source code of PAKDD'25 submission
"TiGer: Self-Supervised Purification for Time-evolving Graphs"


## Environments
- python==3.10.15
- numpy==1.21.2
- torch==2.0.1 (with CUDA 11.8)
- scikit-learn==1.5.1
- scipy==1.13.1
- pyg==2.5.2
- networkx==3.3

## Run TiGer

example: school dataset / random seed: 0 (0 ~ 9)

python TiGer.py --graph_name school --seed 0
