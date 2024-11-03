import numpy as np
import scipy as sp
import scipy.linalg
from itertools import permutations
import networkx as nx
import warnings

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
        self.edge_list = E
        self.adjacency_tensor = A
        self.incidence_matrix = IM
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

    @property
    def num_nodes(self):
        """Returns the number of nodes (vertices) in the hypergraph.

        Returns
        -------
        int
            Number of nodes in the hypergraph.
        """
        return self.nodes.shape[0] if self.nodes is not None else 0

    @property
    def num_edges(self):
        """Returns the number of edges (hyperedges) in the hypergraph.

        Returns
        -------
        int or bool
            Number of edges if available, otherwise False if edges are not defined.
        """
        if self.edge_list is not None:
            return len(self.edge_list)
        elif self.incidence_matrix is not None:
            return self.incidence_matrix.shape[1]
        else:
            return 0

    @property
    def incidence_matrix(self):
        """Returns the incidence matrix of the hypergraph, constructing it if necessary."""
        if self._incidence_matrix is None:
            self.set_incidence_matrix()  # Automatically set if not yet defined
        return self.incidence_matrix

    @incidence_matrix.setter
    def incidence_matrix(self, IM):
        """Sets the incidence matrix for the hypergraph."""
        self.incidence_matrix = IM

    @property
    def edge_list(self):
        """Returns the edge list of the hypergraph, constructing it if necessary."""
        if self._edge_list is None:
            self.set_edge_list()  # Automatically set if not yet defined
        return self.edge_list

    @edge_list.setter
    def edge_list(self, EL):
        """Sets the edge list for the hypergraph."""
        self.edge_list = EL

    @property
    def adjacency_tensor(self):
        """Returns the adjacency tensor of the hypergraph, constructing it if necessary."""
        if self._adjacency_tensor is None:
            self.set_adjacency_tensor()  # Automatically set if not yet defined
        return self.adjacency_tensor

    @adjacency_tensor.setter
    def adjacency_tensor(self, AT):
        """Sets the adjacency tensor for the hypergraph."""
        self.adjacency_tensor = AT


    def set_incidence_matrix(self, IM=None):
        """
        Constructs and sets the incidence matrix for the hypergraph based on either a provided incidence matrix, 
        an edge list, or an adjacency tensor. 
    
        This method creates a binary incidence matrix (`self.incidence_matrix`) that represents connections 
        between nodes and hyperedges. The incidence matrix is a 2D array where each entry (i, j) is 1 if node i 
        is part of hyperedge j, and 0 otherwise. This matrix can be provided directly or derived from an edge 
        list (`self.edge_list`) or an adjacency tensor (`self.adjacency_tensor`). If an incidence matrix is 
        provided as an argument, it will override any previously set matrix.
    
        Parameters
        ----------
        IM : ndarray, optional
            A predefined incidence matrix to set directly. If this is provided, it will be used as the 
            incidence matrix for the hypergraph, and any existing incidence matrix or data will not be recalculated.
    
        Attributes Updated
        ------------------
        self.incidence_matrix : ndarray
            The constructed or provided incidence matrix, stored as a 2D array of shape 
            (number of nodes, number of hyperedges).
    
        Raises
        ------
        Warning
            If `self.incidence_matrix` is already set, or if neither an edge list nor an adjacency tensor 
            is provided to construct the incidence matrix.
    
        Notes
        -----
        This method requires that either `self.edge_list` or `self.adjacency_tensor` is set if `IM` is not provided.
        If both are available, `self.edge_list` takes precedence over `self.adjacency_tensor`.
    
        Examples
        --------
        >>> hypergraph = Hypergraph()
        >>> hypergraph.edge_list = [[0, 1], [1, 2, 3], [0, 3]]  # List of hyperedges
        >>> hypergraph.set_incidence_matrix()
        >>> print(hypergraph.incidence_matrix)
        [[1 0 1]
         [1 1 0]
         [0 1 0]
         [0 1 1]]
        """
        if self.incidence_matrix is not None:
            warning.warn("The Incidence matrix is already set.")
        elif IM is not None:
            self.incidence_matrix = IM

        elif self.edge_list is not None:
            # Initialize the incidence matrix with zeros
            self.incidence_matrix = np.zeros((self.num_nodes(), len(self.edge_list)), dtype=int)
            
            # Fill the incidence matrix
            for i, edge in enumerate(self.edge_list):
                for node in edge:
                    self.incidence_matrix[node, i] = 1
                    
        elif self.adjacency_tensor is not None:
            # Identify the non-zero entries, which represent hyperedges
            hyperedges = np.argwhere(self.adjacency_tensor != 0)
            self.incidence_matrix = np.zeros((self.num_nodes(), len(hyperedges)), dtype=int)
            
            # Populate the incidence matrix
            for k, edge in enumerate(hyperedges):
                for node in edge:
                    incidence_matrix[node, k] = 1  # Node participates in hyperedge `k`
        else:
            warning.warn("Cannot set the incidence matrix without more information.")


    def set_edge_list(self, EL=None):
        """
        Constructs and sets the edge list for the hypergraph based on either a provided edge list, 
        an incidence matrix, or an adjacency tensor.

        This method creates an edge list (`self.edge_list`) representation of the hypergraph, where 
        each entry in the list is a collection of nodes that form a hyperedge. The edge list can be 
        provided directly, or derived from an incidence matrix (`self.incidence_matrix`) or an adjacency 
        tensor (`self.adjacency_tensor`). If an edge list is provided as an argument, it will override 
        any previously set data.

        Parameters
        ----------
        EL : list of list of int, optional
            A predefined edge list to set directly. If provided, it will be used as the hypergraph's 
            edge list, and any existing data will not be recalculated.

        Attributes Updated
        ------------------
        self.edge_list : list of list of int
            A list where each sub-list represents a hyperedge as a list of node indices that form the 
            hyperedge.

        Raises
        ------
        Warning
            If `self.edge_list` is already set, or if neither an incidence matrix nor an adjacency tensor 
            is provided to construct the edge list.

        Notes
        -----
        This method requires that either `self.incidence_matrix` or `self.adjacency_tensor` is set 
        if `EL` is not provided. If both are available, `self.incidence_matrix` takes precedence over 
        `self.adjacency_tensor`.
        """
        if self.edge_list is not None:
            warnings.warn("The edge list is already set.")
        elif EL is not None:
            self.edge_list = EL
        elif self.incidence_matrix is not None:
            # Create edge list from incidence matrix
            self.edge_list = []
            for col in range(self.incidence_matrix.shape[1]):
                edge = np.where(self.incidence_matrix[:, col] != 0)[0].tolist()
                self.edge_list.append(edge)
        elif self.adjacency_tensor is not None:
            # Create edge list from adjacency tensor
            self.edge_list = []
            hyperedges = np.argwhere(self.adjacency_tensor != 0)
            for edge in hyperedges:
                self.edge_list.append(edge.tolist())
        else:
            warnings.warn("Cannot set the edge list without more information.")


    def set_adjacency_tensor(self, AT=None):
        """
        Constructs and sets the adjacency tensor for a k-uniform hypergraph based on either a provided 
        adjacency tensor, an incidence matrix, or an edge list.

        This method creates a k-dimensional adjacency tensor (`self.adjacency_tensor`) to represent the 
        hypergraph. For a k-uniform hypergraph, each entry in this k-dimensional tensor is 1 if the 
        corresponding combination of nodes forms a hyperedge, and 0 otherwise. The adjacency tensor can be 
        provided directly or derived from the incidence matrix (`self.incidence_matrix`) or the edge list 
        (`self.edge_list`). If an adjacency tensor is provided as an argument, it will override any previously set data.

        Parameters
        ----------
        AT : ndarray, optional
            A predefined adjacency tensor to set directly. If provided, it will be used as the hypergraph's 
            adjacency tensor, and any existing data will not be recalculated.

        Attributes Updated
        ------------------
        self.adjacency_tensor : ndarray
            The constructed or provided adjacency tensor, with a shape of (num_nodes, num_nodes, ..., num_nodes)
            with `k` dimensions, where each entry represents a k-node hyperedge with a non-zero value.

        Raises
        ------
        Warning
            If `self.adjacency_tensor` is already set, or if the hypergraph is non-uniform, or if neither 
            an incidence matrix nor an edge list is provided to construct the adjacency tensor.

        Notes
        -----
        This method requires that the hypergraph be k-uniform. If the hypergraph is non-uniform, 
        an adjacency tensor cannot be constructed. This method also requires either 
        `self.incidence_matrix` or `self.edge_list` to be set if `AT` is not provided.
        """
        if not self.uniform:
            warnings.warn("Cannot set an adjacency tensor for a nonuniform hypergraph.")
            return
        else:
            k = self.k

        if self.adjacency_tensor is not None:
            warnings.warn("The adjacency tensor is already set.")
        elif AT is not None:
            self.adjacency_tensor = AT
        elif self.incidence_matrix is not None:
            # Create adjacency tensor from incidence matrix for k-uniform hypergraph
            num_nodes = self.incidence_matrix.shape[0]
            self.adjacency_tensor = np.zeros((num_nodes,) * k, dtype=int)
            num_edges = self.incidence_matrix.shape[1]

            # Populate adjacency tensor based on incidence matrix
            for i in range(num_edges):
                nodes = np.where(self.incidence_matrix[:, i] == 1)[0]
                if len(nodes) == k:
                    # Use multi-dimensional indexing to set entries for the k-node hyperedge
                    self.adjacency_tensor[tuple(nodes)] = 1
                else:
                    warnings.warn(f"Edge {i} does not have exactly {k} nodes; skipping.")

        elif self.edge_list is not None:
            # Create adjacency tensor from edge list for k-uniform hypergraph
            num_nodes = self.num_nodes()
            self.adjacency_tensor = np.zeros((num_nodes,) * k, dtype=int)

            # Populate adjacency tensor based on edge list
            for i, edge in enumerate(self.edge_list):
                if len(edge) == k:
                    # Use multi-dimensional indexing to set entries for the k-node hyperedge
                    self.adjacency_tensor[tuple(edge)] = 1
                else:
                    warnings.warn(f"Edge {i} does not have exactly {k} nodes; skipping.")
        else:
            warnings.warn("Cannot set the adjacency tensor without more information.")

    
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


