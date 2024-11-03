import numpy as np
import scipy as sp
import scipy.linalg
from itertools import permutations
import networkx as nx

import HAT.multilinalg as mla
import HAT.draw
import HAT.HAT

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

    
    def matrixEntropy(self, type='Rodriguez'):
        """Computes hypergraph entropy based on the eigenvalues values of the Laplacian matrix.

        :param type: Type of hypergraph Laplacian matrix. This defaults to 'Rodriguez' and other
            choices inclue ``Bolla`` and ``Zhou`` (See: ``laplacianMatrix()``).
        :type type: *str, optional*
        :return: Matrix based hypergraph entropy
        :rtype: *float*

        Matrix entropy of a hypergraph is defined as the entropy of the eigenvalues of the
        hypergraph Laplacian matrix [1]. This may be applied to any version of the Laplacian matrix.

        References
        ==========
        .. [1] C. Chen and I. Rajapakse, Tensor Entropy for Uniform Hypergraphs, IEEE TRANSACTIONS ON NETWORK SCIENCE AND ENGINEERING (2020)
            (Equation 1) https://arxiv.org/pdf/1912.09624.pdf
        """
        # Auth: Joshua Pickard
        #       jpic@umich.edu
        # Date: Nov 30, 2022
        L = self.laplacianMatrix(type)
        U, V = np.linalg.eig(L)
        return sp.stats.entropy(U)
    
    def avgDistance(self):
        """Computes the average pairwise distance between any 2 vertices in the hypergraph.

        :return: avgDist
        :rtype: *float*

        The hypergraph is clique expanded to a graph object, and the average shortest path on
        the clique expanded graph is returned.
        """
        # Auth: Joshua Pickard
        #       jpic@umich.edu
        # Date: Nov 30, 2022
        G = self.cliqueGraph()
        avgDist = nx.average_shortest_path_length(G)
        return avgDist
    
    def clusteringCoef(self):
        """Computes clustering average clustering coefficient of the hypergraph.

        :return: average clustering coefficient
        :rtype: *float*

        For a uniform hypergraph, the clustering coefficient of a vertex :math:`v_i`
        is defined as the number of edges the vertex participates in (i.e. :math:`deg(v_i)`) divided
        by the number of :math:`k-`way edges that could exist among vertex :math:`v_i` and its neighbors 
        (See equation 31 in [1]). This is written

        .. math::
            C_i = \\frac{deg(v_i)}{\\binom{|N_i|}{k}}

        where :math:`N_i` is the set of neighbors or vertices adjacent to :math:`v_i`. The hypergraph
        clustering coefficient computed here is the average clustering coefficient for all vertices,
        written

        .. math::
            C=\sum_{i=1}^nC_i

        References
        ==========
        .. [1] Surana, Amit, Can Chen, and Indika Rajapakse. "Hypergraph Similarity Measures."
            IEEE Transactions on Network Science and Engineering (2022). 
            https://drive.google.com/file/d/1JUYIQ2_u9YX7ky0U7QptUbJyjEMSYNNR/view
        """
        # Auth: Joshua Pickard
        #       jpic@umich.edu
        # Date: Nov 30, 2022
        n, e = self.IM.shape
        order = sum(self.IM[:,0])
        avgClusterCoef = 0
        for v in range(n):
            edges = np.where(self.IM[v,:] > 0)[0]
            neighbors = np.where(np.sum(self.IM[:, edges], axis=1) > 0)[0]
            avgClusterCoef += (len(edges) / sp.special.comb(len(neighbors), order))
        avgClusterCoef /= n
        return avgClusterCoef
                                   
    def centrality(self, tol=1e-4, maxIter=3000, model='LogExp', alpha=10):
        """Computes node and edge centralities.

        :param tol: threshold tolerance for the convergence of the centrality measures, defaults to 1e-4
        :type tol: *int*, optional
        :param maxIter: maximum number of iterations for the centrality measures to converge in, defaults to 3000
        :type maxIter: int, optional
        :param model: the set of functions used to compute centrality. This defaults to 'LogExp', and other choices include
            'Linear', 'Max' or a list of 4 custom function handles (See [1]).
        :type model: str, optional
        :param alpha: Hyperparameter used for computing centrality (See [1]), defaults to 10
        :type alpha: int, optional

        :return: vxcCentrality
        :rtype: *ndarray* containing centrality scores for each vertex in the hypergraph
        :return: edgeCentrality
        :rtype: *ndarray* containing centrality scores for each edge in the hypergraph
        
        References
        ----------
        .. [1] Tudisco, F., Higham, D.J. Node and edge nonlinear eigenvector centrality for hypergraphs. Commun Phys 4, 201 (2021). https://doi.org/10.1038/s42005-021-00704-2
        """
        # Auth: Joshua Pickard
        #       jpic@umich.edu
        # Date: Nov 30, 2022
        if model=='Linear':
            f = lambda x : x
            g = lambda x : x
            phi = lambda x : x
            psi = lambda x : x
        elif model=='LogExp':
            f = lambda x : x
            g = lambda x : np.sqrt(x)
            phi = lambda x : np.log(x)
            psi = lambda x : np.exp(x)
        elif model=='Max':
            f = lambda x : x
            g = lambda x : x
            phi = lambda x : x**alpha
            psi = lambda x : x**(1/alpha)
        elif isinstance(model, list):
            f = model[0]
            g = model[1]
            phi = model[2]
            psi = model[3]
        else:
            print('Enter a valid centrality model')
            
        B = self.IM
        W = self.edgeWeights
        N = self.nodeWeights
        W = np.diag(W)
        N = np.diag(N)
        n, m = B.shape        
        x0 = np.ones(n,) / n
        y0 = np.ones(m,) / m
        
        for i in range(maxIter):
            u = np.sqrt(np.multiply(np.array(x0),np.array(g(B @ W @ f(y0)))))
            v = np.sqrt(np.multiply(np.array(y0),np.array(psi(B.T @ N @ phi(x0)))))
            x = u / sum(u)
            y = v / sum(v)
            
            check = sum(np.abs(x - x0)) + sum(np.abs(y - y0))
            if check < tol:
                break
            else:
                x0 = x
                y0 = y
                
        vxcCentrality = x
        edgeCentrality = y
        
        return vxcCentrality, edgeCentrality
