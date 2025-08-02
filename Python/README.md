# Hypergraph Class Documentation

## Overview

The `Hypergraph` class provides a flexible representation for undirected and directed hypergraphs, supporting multiple internal representations and automatic conversion between them.

## Constructor

```python
Hypergraph(edge_list=None, adjacency_tensor=None, incidence_matrix=None, 
           nodes=None, edges=None, uniform=None, order=None, directed=None, 
           compress=True, verbose=False)
```

### Parameters

- **edge_list** (*list of lists, optional*): List of hyperedges. For undirected: `[[1,2,3], [2,4,5]]`. For directed: `[[[1], [2,3]], [[4], [5,6]]]` where first sublist is head, second is tail.
- **adjacency_tensor** (*ndarray, optional*): Tensor representation of the hypergraph
- **incidence_matrix** (*ndarray, optional*): Matrix where rows=nodes, columns=edges
- **nodes** (*DataFrame, optional*): Node metadata with required 'Nodes' column
- **edges** (*DataFrame, optional*): Edge metadata with required 'Nodes' column
- **uniform** (*bool, optional*): Whether all edges have the same size
- **order** (*int, optional*): Size of edges (for uniform hypergraphs)
- **directed** (*bool, optional*): Whether hypergraph is directed
- **compress** (*bool, default=True*): Remove duplicate edges during construction
- **verbose** (*bool, default=False*): Enable warning messages

**Note:** At least one of `edge_list`, `adjacency_tensor`, `incidence_matrix`, or `edges` must be provided.

## Key Properties

### Graph Structure
- **nodes**: DataFrame containing node information
- **edges**: DataFrame containing edge information  
- **nnodes**: Number of nodes
- **nedges**: Number of edges
- **directed**: Whether the hypergraph is directed
- **uniform**: Whether all edges have the same size
- **order**: Edge size (for uniform hypergraphs, -1 otherwise)

### Representations
The class automatically converts between three representations as needed:
- **incidence_matrix**: Nodes × edges matrix
- **edge_list**: List of node lists per edge
- **adjacency_tensor**: Tensor representation

### Derived Graphs
- **star_graph**: NetworkX graph where each hyperedge becomes a star
- **clique_graph**: NetworkX graph where each hyperedge becomes a clique
- **connected_components**: Node connectivity information
- **dual**: Dual hypergraph (nodes↔edges swapped)

### Weights
- **node_weights**: Node weights (defaults to 1.0)
- **edge_weights**: Edge weights (defaults to 1.0)

## Methods

### Adding Elements
```python
add_node(properties=None)
add_edge(nodes, properties=None)
```

### Format Conversion
```python
# Class methods for construction
Hypergraph.from_networkx(nxg)
Hypergraph.from_hif(hif_dict)

# Instance methods for export
to_hif()  # Returns HIF dictionary
```

### Properties for Export
- **hif**: Exports to Hypergraph Interchange Format
- **hypergraphx**: Exports to HyperGraphX format

## Examples

### Basic Usage
```python
# Create from edge list
edges = [[0, 1, 2], [1, 2, 3], [2, 3, 4]]
hg = Hypergraph(edge_list=edges)

# Access properties
print(f"Nodes: {hg.nnodes}, Edges: {hg.nedges}")
print(f"Uniform: {hg.uniform}, Order: {hg.order}")

# Get incidence matrix
I = hg.incidence_matrix
```

### Directed Hypergraph
```python
# Directed edges: [head_nodes, tail_nodes]
directed_edges = [[[0], [1, 2]], [[1], [2, 3]]]
hg = Hypergraph(edge_list=directed_edges, directed=True)
```

### With Metadata
```python
import pandas as pd

# Node metadata
nodes_df = pd.DataFrame({
    'Nodes': [0, 1, 2, 3],
    'label': ['A', 'B', 'C', 'D'],
    'weight': [1.0, 2.0, 1.5, 1.0]
})

# Edge metadata  
edges_df = pd.DataFrame({
    'Nodes': [[0, 1], [1, 2, 3]],
    'weight': [1.0, 2.0],
    'type': ['strong', 'weak']
})

hg = Hypergraph(nodes=nodes_df, edges=edges_df)
```

## Implementation Notes

- **Automatic Detection**: The class automatically detects whether the hypergraph is uniform, directed, etc.
- **Lazy Evaluation**: Representations are computed only when accessed
- **Node Conversion**: String node names are automatically converted to integers internally
- **Duplicate Removal**: When `compress=True`, duplicate edges are automatically removed

## Related Functions

- `HAT.graph.star_graph()` - Convert to star expansion
- `HAT.graph.clique_graph()` - Convert to clique expansion  
- `HAT.laplacian.laplacian_matrix()` - Compute Laplacian matrix
- `HAT.export.to_hif()` - Export to HIF format
- `HAT.export.to_hypergraphx()` - Export to HyperGraphX format

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