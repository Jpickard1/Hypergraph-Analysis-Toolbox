# Hypergraph Analysis Toolbox (HAT): Python Implementation

---

This directory manages the Python implenentation of HAT.

## Hypergraph Class:

**Numerical Representation**
- edge_list
- adjacency_tensor
- incidence_matrix

**Annotated Features**
- nodes
- edges

**Hyeprgraph Attributes**
- uniform
- order
- directed

**Construction**

*Criteria to be directed*
- We assume adjacency tensors are directed
- Signed incidence matrices are directed, else undirected
- head tail edge lists are directed

### Data Schema:

**Numerical Representation**
- edge_list:
```
# Undirected
HG.edge_list = [
    [0,1,2],
    [0,1,3]
]

# Directed
HG.edge_list = [
    [[head], [tail]],
    [[0],[1,2]],
    [[0],[1,3]]
]
```

- incidence_matrix
```
# Undirected
HG.incidence_matrix = np.array(
    [[1, 1],
     [1, 1],
     [1, 0],
     [0, 1]
    ]
)
# Directed
HG.incidence_matrix = np.array(
    [[ 1,  1],
     [-1, -1],
     [-1,  0],
     [ 0, -1]
    ]
)
```
- adjacency_tensor (only directed)
```
# Directed
HG.adjacency_tensor = np.array(
    [
        [
            [0, 0, 0, 0],
            [0, 0, 1, 1],
            [0, 1, 0, 0],
            [0, 1, 0, 0]
        ],
        [
            [0, 0, 1, 1],
            [0, 0, 0, 0],
            [1, 0, 0, 0],
            [1, 0, 0, 0]
        ],
        [
            [0, 1, 0, 0],
            [1, 0, 0, 0],
            [0, 0, 0, 0],
            [0, 0, 0, 0]
        ],
        [
            [0, 1, 0, 0],
            [1, 0, 0, 0],
            [0, 0, 0, 0],
            [0, 0, 0, 0]
        ]
    ]
)
```

**Annotated Features**
- nodes
- edges

**Hyeprgraph Attributes**
- uniform (bool): True if HG is *k-*uniform (undirected) or *k-*tail-uniform (directed)
- order (int, None): the order of the uniformity
- directed (bool): if hyperedges have direction associated with them
- weighted (bool): if hyperedges have associated weights


## TODO:

1. decide to remove setters of incidence matrix, adjacency tensors, and edge sets.
    Argument for: this should be supplied at creation or by modifiation of adding new vertices. If it is already set, why would we reset it?
    Argument against: not sure.

2. Testing:

```
python -m unittest -v .\tests\hypergraph_constructors.py
```