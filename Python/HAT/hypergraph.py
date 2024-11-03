import numpy as np
import scipy as sp
import scipy.linalg
from itertools import permutations
import networkx as nx

import multilinalg as mla
import draw
import HAT

class Hypergraph:
    """Represents a hypergraph structure, enabling complex multi-way relationships between nodes.

    This class implements a hypergraph where edges (or hyperedges) can connect multiple vertices. 
    Internally, the hypergraph can be represented by an incidence matrix, an adjacency tensor, or an 
    edge list, depending on the available inputs and the uniformity of the hypergraph.

    Formally, a hypergraph :math:`H=(V,E)` is defined by a set of vertices :math:`V` and a set of 
    edges :math:`E`, where each edge :math:`e \\in E` may contain any subset of vertices in :math:`V`. 
    Unlike traditional graphs, hypergraph edges can connect more than two vertices, allowing for 
    multi-way interactions. Uniform hypergraphs, where all edges have the same number of vertices, 
    can be efficiently represented using tensors.

    Parameters
    ----------
    E : list of lists, optional
        List of edges where each edge is represented by a list of node indices. 
        Each list element corresponds to an edge.
    A : ndarray, optional
        Adjacency tensor of the hypergraph, primarily used if the hypergraph is uniform.
    IM : ndarray, optional
        Incidence matrix representing node-edge connections, where rows represent nodes 
        and columns represent edges.
    nodes : ndarray, optional
        Array of nodes or vertices in the hypergraph.
    edges : ndarray, optional
        Array of edges (or hyperedges), where each entry is a collection of nodes.
    uniform : bool, optional
        Whether the hypergraph is uniform (all edges contain the same number of vertices).
        If not specified, it will be inferred from other inputs.
    k : int, optional
        The edge degree for uniform hypergraphs (i.e., the number of nodes per edge).
    directed : bool, default=False
        Indicates if the hypergraph is directed.

    Attributes
    ----------
    edgelist : list of lists or None
        Stores the list of edges if provided during initialization.
    adj_tensor : ndarray or None
        Stores the adjacency tensor if provided during initialization.
    IM : ndarray or None
        Stores the incidence matrix if generated or provided during initialization.
    nodes : ndarray or None
        Array of nodes (vertices) in the hypergraph.
    edges : ndarray or None
        Array of edges (hyperedges) in the hypergraph.
    uniform : bool
        Indicates if the hypergraph is uniform.
    k : int
        The degree of uniformity in terms of edge size, if applicable.
    """
    def __init__(self, E=None, A=None, IM=None, nodes=None, edges=None, uniform=None, k=None, directed=False):
        # Auth: Joshua Pickard
        #       jpic@umich.edu
        # Date: Nov 30, 2022
        # If all E, A, and IM are None, display a warning
        if E is None and A is None and IM is None:
            warnings.warn("Warning: All of E, A, and IM are None.", UserWarning)

        # Assigning instance attributes
        self.edgelist = E
        self.adj_tensor = A
        self.IM = IM
        self.nodes = nodes
        self.edges = edges
        self.uniform = uniform
        self.k = -1

        # Determine if the hypergraph is uniform
        if self.uniform is None:
            if self.adj_tensor is not None:
                self.uniform = True
                self.k = len(self.adj_tensor.shape)
            elif self.IM is not None:
                nonzero_counts = np.count_nonzero(arr, axis=0)
                self.uniform = np.all(nonzero_counts == nonzero_counts[0])
                self.k = nonzero_counts[0]
            elif self.edgelist is not None:
                k = len(self.edgelist[0])
                for e in self.edgelist:
                    if len(e) != k:
                        self.uniform = False
                        break
                if self.uniform is None:
                    self.k = k
                    self.uniform = True

    def num_nodes(self):
        """Returns the number of nodes (vertices) in the hypergraph.

        Returns
        -------
        int
            Number of nodes in the hypergraph.
        """
        return self.nodes.shape[0]

    def num_edges(self):
        """Returns the number of edges (hyperedges) in the hypergraph.

        Returns
        -------
        int or bool
            Number of edges if available, otherwise False if edges are not defined.
        """
        if self.edgelist is not None:
            return len(self.edgelist)
        elif self.IM is not None:
            return self.IM.shape[1]
        else:
            return False
    
    def set_IM(self):
        """Constructs the incidence matrix (IM) from the edge list (E).

        This method creates a binary incidence matrix representing connections 
        between nodes and edges, where each element at (i, j) is 1 if node i 
        is part of edge j, and 0 otherwise. This method requires that an 
        edge list is already provided in `self.edgelist`.
        
        Updates
        -------
        self.IM : ndarray
            The incidence matrix after construction based on `self.edgelist`.
        """
        if self.edgelist is not None:
            # Initialize the incidence matrix with zeros
            self.IM = np.zeros((self.num_nodes(), len(self.edgelist)), dtype=int)
            
            # Fill the incidence matrix
            for i, edge in enumerate(self.edgelist):
                for node in edge:
                    self.IM[node, i] = 1
    
    def draw(self, shadeRows=True, connectNodes=True, dpi=200, edgeColors=None):
        """ This function draws the incidence matrix of the hypergraph object. It calls the function
        ``HAT.draw.incidencePlot``, but is provided to generate the plot directly from the object.

        :param shadeRows: shade rows (bool)
        :param connectNodes: connect nodes in each hyperedge (bool)
        :param dpi: the resolution of the image (int)
        :param edgeColors: The colors of edges represented in the incidence matrix. This is random by default
        
        :return: *matplotlib* axes with figure drawn on to it
        """
        # Auth: Joshua Pickard
        #       jpic@umich.edu
        # Date: Nov 30, 2022
        return HAT.draw.incidencePlot(self, shadeRows=shadeRows, connectNodes=connectNodes, dpi=dpi, edgeColors=edgeColors)
        
    def dual(self):
        """The dual hypergraph is constructed.

        :return: Hypergraph object
        :rtype: *Hypergraph*

        Let :math:`H=(V,E)` be a hypergraph. In the dual hypergraph each original edge :math:`e\in E`
        is represented as a vertex and each original vertex :math:`v\in E` is represented as an edge. Numerically, the
        transpose of the incidence matrix of a hypergraph is the incidence matrix of the dual hypergraph.
        
        References
        ----------
        .. [1] Yang, Chaoqi, et al. "Hypergraph learning with line expansion." arXiv preprint arXiv:2005.04843 (2020).
        """
        # Auth: Joshua Pickard
        #       jpic@umich.edu
        # Date: Nov 30, 2022
        IM = self.IM.T
        return Hypergraph(IM)


