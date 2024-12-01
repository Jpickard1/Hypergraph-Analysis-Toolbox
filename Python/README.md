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