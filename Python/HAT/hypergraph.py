import numpy as np
import scipy as sp
import scipy.linalg
from itertools import permutations
import networkx as nx
import warnings
import pandas as pd

# import multilinalg as mla
# import draw
# import HAT

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
    incidence_matrix : ndarray, optional
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
    edge_list : list of lists or None
        Stores the list of edges if provided during initialization.
    adjacency_tensor : ndarray or None
        Stores the adjacency tensor if provided during initialization.
    incidence_matrix : ndarray or None
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
    def __init__(self,
            edge_list=None,
            adjacency_tensor=None,
            incidence_matrix=None,
            nodes=None,
            edges=None,
            uniform=None,
            order=None,
            directed=None
        ):
        # Auth: Joshua Pickard
        #       jpic@umich.edu
        # Date: Nov 30, 2022

        # Validate arguments
        self._validate_constructor_arguments(
            edge_list        = edge_list,
            adjacency_tensor = adjacency_tensor,
            incidence_matrix = incidence_matrix,
            nodes            = nodes,
            edges            = edges,
            uniform          = uniform,
            order            = order,
            directed         = directed
        )

        # Assign object data
        self._edge_list        = edge_list
        self._adjacency_tensor = adjacency_tensor
        self._incidence_matrix = incidence_matrix
        self._nodes            = nodes
        self._edges            = edges
        self._uniform          = uniform
        self._order            = order
        self._directed         = directed

        # Set uniform and order of the hypergraph
        self._detect_uniform_and_order(uniform, order)

        # Set directed property of the hypergraph
        self._detect_directed(directed)

        # Set the nodes dataframe
        if self._nodes is None:
            if self._adjacency_tensor is not None:
                num_nodes = self._adjacency_tensor.shape[0]
            elif self._edge_list is not None:
                num_nodes = len(list(set([j for edge in edge_list for j in edge])))
            elif self._incidence_matrix is not None:
                num_nodes = self.incidence_matrix.shape[0]
            self._nodes = pd.DataFrame({'Nodes': np.arange(num_nodes)})
        elif 'Nodes' not in self._nodes.columns:
            self._nodes['Nodes'] = np.arange(self._nodes.shape[0])
            warnings.warn('`Nodes` column not found in the provided in self._nodes. It has been added')

        # Set the edges dataframe
        if self._edges is None:
            # Determine the number of edges, the nodes, the head, and the tail
            num_edges, edge_nodes, head, tail = 0, [], [], []
            if self._edge_list is not None:
                num_edges = len(self._edge_list)
                if self._directed:
                    for edge in self._edge_list:
                        edge_nodes.append(edge[0] + edge[1])
                        head.append(edge[0])
                        tail.append(edge[1])
                else:
                    edge_nodes = self._edge_list
            elif self._incidence_matrix is not None:
                num_edges = self.incidence_matrix.shape[1]
                for iedge in range(num_edges):
                    edge_nodes.append(np.where(self._incidence_matrix[:,iedge] != 0)[0])
                    if self._directed:
                        head.append(np.where(self._incidence_matrix[:,iedge] > 0)[0])
                        tail.append(np.where(self._incidence_matrix[:,iedge] < 0)[0])
            elif self._adjacency_tensor is not None:
                idxs = np.where(self._adjacency_tensor != 0) 
                head = list(idxs[0])
                for iedge in range(len(idxs[0])):
                    tail.append([idxs[k][iedge] for k in range(1,len(idxs))])
                    edge_nodes.append(head[iedge] + tail[-1])
                num_edges = len(head)
            if self.directed:
                self._edges = pd.DataFrame(
                    {
                        'Edges' : np.arange(num_edges),
                        'Nodes' : edge_nodes,
                        'Head'  : head,
                        'Tail'  : tail
                    }
                )
            else:
                self._edges = pd.DataFrame(
                    {
                        'Edges' : np.arange(num_edges),
                        'Nodes' : edge_nodes,
                    }
                )
        elif 'Edges' not in self._edges.columns:
            self._edges = pd.DataFrame({'Edges': list(np.arange(self._edges.shape[0]))})
            warnings.warn('"Edges" column not found in the provided nodes dataframe.')
            warnings.warn('This column has been appended.')

        if self._directed:
            if 'Head' not in self._edges.columns:
                self._edges['Head'] = self._edges['Nodes']
            if 'Tail' not in self._edges.columns:
                self._edges['Tail'] = self._edges['Nodes']

    def _validate_constructor_arguments(
        self,
        edge_list=None,
        adjacency_tensor=None,
        incidence_matrix=None,
        nodes=None,
        edges=None,
        uniform=None,
        order=None,
        directed=None
    ):
        # At least one numerical representation should be supplied
        if edge_list is None and adjacency_tensor is None and incidence_matrix is None:
            warnings.warn("The edge list, incidence matrix, and adjacency tensor are None.", UserWarning)

        if order is not None and not (uniform == True):
            warnings.warn("If the hypergraph order is fixed then it should be k-uniform")

        # TODO: add more conditions to validate / warn the user of during construction

    def _detect_uniform_and_order(self, uniform, order):
        if self._adjacency_tensor is not None:
            self._uniform = True
            self._order = len(self._adjacency_tensor.shape)
        elif self._incidence_matrix is not None:
            nonzero_counts = np.count_nonzero(self._incidence_matrix, axis=0)
            self._uniform = np.all(nonzero_counts == nonzero_counts[0])
            self._order = nonzero_counts[0]
        elif self._edge_list is not None:
            # Determine if the hypergraph is directed
            if isinstance(self._edge_list[0][0], int): # (undirected)
                k = len(self._edge_list[0])
                self._uniform = True
                self._order = k
                for e in self._edge_list:
                    if len(e) != k:
                        self._uniform = False
                        break
            else:  # (directed)
                k = len(self._edge_list[0][0])
                self._uniform = True
                self._order = k
                for e in self._edge_list:
                    if len(e[0]) != k:
                        self._uniform = False
                        break
        if self._uniform == False:
            self._order = -1

        if self._uniform != uniform and uniform is not None:
            warnings.warn('The provided and detected `uniform` are not in agreement!')

        if self._order != order and order is not None:
            warnings.warn('The provided and detected `order` are not in agreement!')

    def _detect_directed(self, directed):
        if self._edges is not None and 'Head' in self._edges.columns and 'Tail' in self._edges.columns:
            self._directed = True

        # The adjacency tensor is directed
        elif self._adjacency_tensor is not None:
            self._directed = True

        # The indicence matrix is directed if there are +/- terms
        elif self._incidence_matrix is not None and np.sum(self._incidence_matrix > 0) > 0 and np.sum(self._incidence_matrix < 0) > 0:
            self._directed = True

        # The edge list is directed if an edge has the form: [[head], [tail]]
        elif self._edge_list is not None:
            self._directed = True
            for edge in self._edge_list:
                if not (len(edge) >= 2 and isinstance(edge[0], list) and isinstance(edge[1], list)):
                    self._directed = False

        # By default, the system is undirected
        else:
            self._directed = False

        if self._directed != directed and directed is not None:
            warnings.warn('The provided and detected `directed` are not in agreement!')

    @property
    def nodes(self):
        return self._nodes
        
    @property
    def edges(self):
        return self._edges

    @property
    def directed(self):
        return self._directed

    @property
    def nnodes(self):
        """Returns the number of nodes (vertices) in the hypergraph.

        Returns
        -------
        int
            Number of nodes in the hypergraph.
        """
        return self._nodes.shape[0] if self._nodes is not None else 0

    @property
    def nedges(self):
        """Returns the number of edges (hyperedges) in the hypergraph.

        Returns
        -------
        int or bool
            Number of edges if available, otherwise False if edges are not defined.
        """
        if self._edge_list is not None:
            return len(self._edge_list)
        elif self._incidence_matrix is not None:
            return self._incidence_matrix.shape[1]
        else:
            return 0

    @property
    def uniform(self):
        """Returns if the hypergraph was uniform."""
        return self._uniform
    
    @property
    def order(self):
        return self._order
    
    @property
    def incidence_matrix(self):
        """Returns the incidence matrix of the hypergraph, constructing it if necessary."""
        if self._incidence_matrix is None:
            self.set_incidence_matrix()  # Automatically set if not yet defined
        return self._incidence_matrix

#    @incidence_matrix.setter
#    def incidence_matrix(self, IM):
#        """Sets the incidence matrix for the hypergraph."""
#        self._incidence_matrix = IM

    @property
    def edge_list(self):
        """Returns the edge list of the hypergraph, constructing it if necessary."""
        if self._edge_list is None:
            self.set_edge_list()  # Automatically set if not yet defined
        return self._edge_list

#    @edge_list.setter
#    def edge_list(self, EL):
#        """Sets the edge list for the hypergraph."""
#        self._edge_list = EL

    @property
    def adjacency_tensor(self):
        """Returns the adjacency tensor of the hypergraph, constructing it if necessary."""
        if self._adjacency_tensor is None:
            self.set_adjacency_tensor()  # Automatically set if not yet defined
        return self._adjacency_tensor

#    @adjacency_tensor.setter
#    def adjacency_tensor(self, AT):
#        """Sets the adjacency tensor for the hypergraph."""
#        self._adjacency_tensor = AT

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
        if self._incidence_matrix is not None:
            warnings.warn("The Incidence matrix is already set.")

        elif IM is not None:
            self._incidence_matrix = IM

        elif self._edge_list is not None:
            # Initialize the incidence matrix with zeros
            self._incidence_matrix = np.zeros((self.nnodes, len(self.edge_list)), dtype=int)
            
            # Fill the incidence matrix
            for i, edge in enumerate(self.edge_list):
                for node in edge:
                    self._incidence_matrix[node, i] = 1
                    
        elif self._adjacency_tensor is not None:
            # Identify the non-zero entries, which represent hyperedges
            hyperedges = np.argwhere(self._adjacency_tensor != 0)
            self._incidence_matrix = np.zeros((self._nodes.shape[0], hyperedges.shape[0]), dtype=int)
            
            # Populate the incidence matrix
            for k, edge in enumerate(hyperedges):
                for i, node in enumerate(edge):
                    self._incidence_matrix[node, k] = 1  # Node participates in hyperedge `k`
                    if i == 0:
                        self._incidence_matrix[node, k] = -1
        else:
            warning.warn("Cannot set the incidence matrix without more information.")

        # TODO: ensure .edges aligns after this sort
        _, unique_indices = np.unique(self._incidence_matrix, axis=1, return_index=True)
        self._incidence_matrix = self._incidence_matrix[:, np.sort(unique_indices)]

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
        if self._edge_list is not None:
            warnings.warn("The edge list is already set.")
        elif self._directed is False and EL is not None:
            self._edge_list = EL
        elif self._directed is False and self.incidence_matrix is not None:
            # Create edge list from incidence matrix
            self._edge_list = []
            for col in range(self.incidence_matrix.shape[1]):
                edge = np.where(self.incidence_matrix[:, col] != 0)[0].tolist()
                self._edge_list.append(edge)
        # Adjacency tensor assumes it is directed
        elif self._adjacency_tensor is not None:
            # Create edge list from adjacency tensor
            self._edge_list = []
            hyperedges = np.argwhere(self.adjacency_tensor != 0)
            for edge in hyperedges:
                edge_nodes = edge.tolist()
                self._edge_list.append([[edge_nodes[0]], sorted(edge_nodes[1:])])
        else:
            warnings.warn("Cannot set the edge list without more information.")
        self._edge_list = self._remove_duplicates(self._edge_list)

    def _remove_duplicates(self, list_of_lists):
        if isinstance(list_of_lists[0][0], int):
            return list_of_lists
        # Use a set to track unique items
        unique_items = set()
        deduplicated_list = []

        for item in list_of_lists:
            # Convert the two inner lists to a tuple of sorted tuples
            normalized_item = tuple(sorted(map(tuple, item)))
            if normalized_item not in unique_items:
                unique_items.add(normalized_item)
                deduplicated_list.append(item)

        return deduplicated_list

    def set_adjacency_tensor(self, adjacency_tensor=None):
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
        adjacency_tensor : ndarray, optional
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
        if self._adjacency_tensor is not None:
            warnings.warn("The adjacency tensor is already set. Unclear why this is being reset.")

        if not self.uniform:
            warnings.warn("Cannot set an adjacency tensor for a nonuniform hypergraph.")
            return

        if adjacency_tensor is not None:
            self._adjacency_tensor = adjacency_tensor
            # TODO: in this case, we may want to reset the _nodes and _edges.
            #       I am not sure when a user may want to do this
            return

        # If the system is directed, then we set it one way        
        if not self.directed and self._incidence_matrix is not None:
                # Create adjacency tensor from incidence matrix for k-uniform hypergraph
                num_nodes = self.incidence_matrix.shape[0]
                self._adjacency_tensor = np.zeros((num_nodes,) * self.order, dtype=int)
                num_edges = self.incidence_matrix.shape[1]

                # Populate adjacency tensor based on incidence matrix
                for i in range(num_edges):
                    nodes = np.where(self.incidence_matrix[:, i] == 1)[0]
                    if len(nodes) == self.order:
                        # Use multi-dimensional indexing to set entries for the k-node hyperedge
                        for nodes_indices in permutations(nodes):
                            self._adjacency_tensor[tuple(nodes_indices)] = 1
                    else:
                        warnings.warn(f"Edge {i} does not have exactly {self._order} nodes; skipping.")

        elif not self.directed and self.edge_list is not None:
            # Create adjacency tensor from edge list for k-uniform hypergraph
            num_nodes = self.nnodes
            self._adjacency_tensor = np.zeros((num_nodes,) * self.order, dtype=int)
            # Populate adjacency tensor based on edge list
            for i, edge in enumerate(self.edge_list):
                if len(edge) == self.order:
                    # Use multi-dimensional indexing to set entries for the k-node hyperedge
                    for nodes_indices in permutations(edge):
                        self._adjacency_tensor[tuple(nodes_indices)] = 1
                else:
                    warnings.warn(f"Edge {i} does not have exactly {k} nodes; skipping.")

    def dual(self):
        """The dual hypergraph is constructed.
        Let :math:`H=(V,E)` be a hypergraph. In the dual hypergraph each original edge :math:`e in E`
        is represented as a vertex and each original vertex :math:`v in E` is represented as an edge. Numerically, the
        transpose of the incidence matrix of a hypergraph is the incidence matrix of the dual hypergraph.
        
        :return: Hypergraph object
        :rtype: *Hypergraph*

        References
        ----------
        .. [1] Yang, Chaoqi, et al. "Hypergraph learning with line expansion." arXiv preprint arXiv:2005.04843 (2020).
        """
        # Auth: Joshua Pickard
        #       jpic@umich.edu
        # Date: Nov 30, 2022
        IM = self.IM.T
        return Hypergraph(IM)
    
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
        # return HAT.draw.incidencePlot(self, shadeRows=shadeRows, connectNodes=connectNodes, dpi=dpi, edgeColors=edgeColors)
        pass

