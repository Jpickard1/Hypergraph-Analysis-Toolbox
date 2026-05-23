# Hypergraph Analysis Toolbox (HAT)

[![Documentation](https://readthedocs.org/projects/hypergraph-analysis-toolbox/badge/?version=latest)](https://hypergraph-analysis-toolbox.readthedocs.io/en/latest/)
[![MATLAB File Exchange](https://www.mathworks.com/matlabcentral/images/matlab-file-exchange.svg)](https://www.mathworks.com/matlabcentral/fileexchange/121013-hypergraph-analysis-toolbox)
[![PyPI version](https://img.shields.io/pypi/v/HypergraphAnalysisToolbox)](https://pypi.org/project/HypergraphAnalysisToolbox/)

HAT is a general-purpose software suite for constructing, analyzing, and visualizing hypergraphs and higher-order structures. Originally motivated by the analysis of Pore-C genomic data, HAT is designed to be versatile and extensible across domains with a focus on tensor, dynamics, and control
based algorithms for higher order networks

---

## Features

- **Flexible construction** — build hypergraphs from edge lists, incidence matrices, or adjacency tensors; directed and undirected; weighted and unweighted
- **Metrics** — centrality, similarity, entropy, and more
- **Spectral methods** — Laplacians and related operators
- **Tensor analysis** — eigenvalues, decompositions, and Kronecker products
- **Controllability and observability** — analysis of higher-order dynamical systems
- **Visualization** — hypergraph drawing utilities
- **Interoperability** — import/export with HIF, HyperNetX, and HypergraphX

---

## Installation

### Python

```bash
pip install HypergraphAnalysisToolbox
```

Requires Python ≥ 3.11.

### MATLAB

Download from [MATLAB File Exchange](https://www.mathworks.com/matlabcentral/fileexchange/121013-hypergraph-analysis-toolbox) or clone this repository and add the `Matlab/` directory to your MATLAB path.

---

## Quick Start

```python
import numpy as np
from HAT import Hypergraph

# Construct from an edge list
H = Hypergraph(edge_list=[[0, 1, 2], [0, 1, 3]])

# Construct from an incidence matrix
D = np.array([[1, 1],
              [1, 1],
              [1, 0],
              [0, 1]])
H = Hypergraph(incidence_matrix=D)

```

---

## Documentation

Full documentation — including API reference, tutorials, and examples — is available at:

**https://hypergraph-analysis-toolbox.readthedocs.io**

---

## Publications

HAT consolidates and contains methods from the following works:

| Paper | Link |
|---|---|
| Structural Controllability of Large-Scale Hypergraphs | [Preprint](https://arxiv.org/pdf/2603.19955) |
| Data-Driven Tensor Decomposition Identification of Homogeneous Polynomial Dynamical Systems | [Preprint](https://arxiv.org/pdf/2604.03508) |
| Scalable Hypergraph Algorithms for Observability of Gene Regulation | [European Control Conference](https://ieeexplore.ieee.org/document/11186979) |
| Deciphering Multiway Interactions in the Human Genome | [Nature Communications](https://drive.google.com/file/d/1rp6ZtKf_DxUL0xOpcUzRg1rJGA6xlu-f/view) |
| Geometric Aspects of Observability of Hypergraphs | [IFAC LHMNC](https://drive.google.com/file/d/1-5AL_rOvAm-aUClSfyy9MpED7h7_L76o/view) |
| Observability of Hypergraphs | [IEEE CDC](https://drive.google.com/file/d/1FQxRj5VdPkY-P64ek7rq4lp9jQW9MLFP/view) |
| Kronecker Products of Tensors and Hypergraphs | [SIAM Journal on Matrix Analysis and Applications](https://epubs.siam.org/doi/full/10.1137/23M1592547) |
| HAT: Hypergraph Analysis Toolbox | [PLOS Computational Biology](https://journals.plos.org/ploscompbiol/article?id=10.1371/journal.pcbi.1011190) |
| Hypergraph Similarity Measures | [IEEE TRANSACTIONS ON NETWORK SCIENCE AND ENGINEERING](https://drive.google.com/file/d/1Dc4nSkkZyk4axOAshdRDXo8Gx24h4M-d/view) |
| Controllability of Hypergraphs | [IEEE TRANSACTIONS ON NETWORK SCIENCE AND ENGINEERING](https://drive.google.com/file/d/12aReE7mE4MVbycZUxUYdtICgrAYlzg8o/view) |
| Tensor Entropy for Uniform Hypergraphs | [Tensor Entropy for Uniform Hypergraphs](https://drive.google.com/file/d/1-d4uR5KT3iDpOd69aCQVjNZzMyILXp5E/view) |
| Multilinear Control Systems Theory | [SIAM J. CONTROL OPTIM.](https://drive.google.com/file/d/1F0ZGoVWeKSWemXvSp6ilTwLS8N7j6n71/view) |

---

## Citation

If you use HAT in your research, please cite:

```bibtex
@article{pickard2023hat,
    title={HAT: Hypergraph analysis toolbox},
    author={Pickard, Joshua and Chen, Can and Salman, Rahmy and Stansbury, Cooper and Kim, Sion and Surana, Amit and Bloch, Anthony and Rajapakse, Indika},
    journal={PLOS Computational Biology},
    volume={19},
    number={6},
    pages={e1011190},
    year={2023},
    publisher={Public Library of Science San Francisco, CA USA}
}
```

---

## Contributing

Bug reports and feature requests are welcome via [GitHub Issues](https://github.com/Jpickard1/Hypergraph-Analysis-Toolbox/issues).
