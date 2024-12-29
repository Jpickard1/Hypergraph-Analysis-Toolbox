import numpy as np
import scipy as sp
import networkx as nx

"""
This file interfaces HAT with networkx. It provides several methods to create graph representations from hypergraphs.
"""


def clique_graph(HG):
    """The clique expansion graph is constructed.

    :return: Clique expanded graph
    :rtype: *networkx.graph*

    The clique expansion algorithm constructs a *graph* on the same set of vertices as the hypergraph by defining an
    edge set where every pair of vertices contained within the same edge in the hypergraph have an edge between them
    in the graph. Given a hypergraph :math:`H=(V,E_h)`, then the corresponding clique graph is :math:`C=(V,E_c)` where
    :math:`E_c` is defined

    .. math::

        E_c = \{(v_i, v_j) |\ \exists\  e\in E_h \\text{ where } v_i, v_j\in e\}.

    This is called clique expansion because the vertices contained in each :math:`h\in E_h` forms a clique in :math:`C`.
    While the map from :math:`H` to :math:`C` is well-defined, the transformation to a clique graph is a lossy process,
    so the hypergraph structure of :math:`H` cannot be uniquely recovered from the clique graph :math:`C` alone [1].

    References
    ----------
    .. [*] Amit Surana, Can Chen, and Indika Rajapakse. Hypergraph similarity measures. IEEE Transactions on Network Science and Engineering, pages 1-16, 2022.
    .. [*] Yang, Chaoqi, et al. "Hypergraph learning with line expansion." arXiv preprint arXiv:2005.04843 (2020).
    """
    # Auth: Joshua Pickard
    #       jpic@umich.edu
    # Date: Nov 30, 2022

    A = HG.incidence_matrix @ HG.incidence_matrix.T
    np.fill_diagonal(A,0)         # Omit self loops
    return nx.from_numpy_array(A)

def lineGraph(HG):
    """The line graph, which is the clique expansion of the dual graph, is constructed.
    
    :return: Line graph
    :rtype: *networkx.graph*

    References
    ----------
    .. [1] Yang, Chaoqi, et al. "Hypergraph learning with line expansion." arXiv preprint arXiv:2005.04843 (2020).
    """
    # Auth: Joshua Pickard
    #       jpic@umich.edu
    # Date: Nov 30, 2022
    D = HG.dual()
    return D.cliqueGraph()

def star_graph(HG):
    """The star graph representation is constructed.

    :return: Star graph
    :rtype: *networkx.graph*

    The star expansion of :math:`{H}=({V},{E}_h)` constructs a bipartite graph :math:`{S}=\{{V}_s,{E}_s\}`
    by introducing a new set of vertices :math:`{V}_s={V}\cup {E}_h` where some vertices in the star graph
    represent hyperedges of the original hypergraph. There exists an edge between each vertex :math:`v,e\in {V}_s`
    when :math:`v\in {V}`, :math:`e\in {E}_h,` and :math:`v\in e`. Each hyperedge in :math:`{E}_h` induces
    a star in :math:`S`. This is a lossless process, so the hypergraph structure of :math:`H` is well-defined]
    given a star graph :math:`S`.
    
    References
    ----------
    .. [1] Yang, Chaoqi, et al. "Hypergraph learning with line expansion." arXiv preprint arXiv:2005.04843 (2020).
    """
    # Auth: Joshua Pickard
    #       jpic@umich.edu
    # Date: Nov 30, 2022
    n = len(HG.incidence_matrix) + len(HG.incidence_matrix[0])
    A = np.zeros((n,n))
    A[len(A) - len(HG.incidence_matrix):len(A),0:len(HG.incidence_matrix[0])] = HG.incidence_matrix
    A[0:len(HG.incidence_matrix[0]),len(A) - len(HG.incidence_matrix):len(A)] = HG.incidence_matrix.T
    return nx.from_numpy_array(A)


class graph:
    """This class represents pairwise graph structures.

    .. warning::
        It is not clear what this class was designed to do or ever fully implemented.
        
    """
    def __init__(self, a=None):
        """Constructs a default graph.
        """
        self.A = a
    
    @property
    def degree(self):
        """The degree matrix of a graph.
        """
        return np.diag(sum(self.A))
        
    @property
    def laplacian(self):
        """The Laplacian matrix of a graph. See equation 1 in [1].
        
        References
        ----------
        .. [1] Surana, Amit, Can Chen, and Indika Rajapakse. "Hypergraph dissimilarity measures." arXiv preprint arXiv:2106.08206 (2021).
        """
        return self.degree - self.A
    
    @property
    def normLaplacian(self):
        """The normalized Laplacian matrix of a graph. See chapter 16.3.3 [1].
        
        References
        ----------
        .. [1] Spielman, Daniel. "Spectral graph theory." Combinatorial scientific computing 18 (2012).
        """
        D = np.matrix_power(self.D, -0.5)
        return D @ self.L @ D


    @property
    def clusteringCoef(self):
        """The clustering coefficient of the graph.
        """
        gammas = 0
        for vx in range(len(self.A)):
            gammas += self.clusteringCoef(vx)
        return gammas / len(self.A)
        
    def clusteringCoef(self, vx):
        """Computes the clustering coefficient of a single vertex.
        """
        neighbors = np.where(self.A[vx,:] == 1)[0]
        if vx not in neighborhood:
            neighbors = np.append(neighbors, vx)
        neighborhood = self.A[neighbors, :]
        neighborhood = neighborhood[:, neighbors]
        realEdges = sum(neighborhood) / 2
        possibleEdges = sp.special.binom(len(neighbors), 2)
        return realEdges / possibleEdges

    def pairwiseDistance(self, vxi, vxj):
        """Computes the pairwise distance between vertices i and j.
        """
        d = 1
        N = self.A
        while N[vxi,vxj] == 0:
            N = np.matmul(N, self.A)
            d += 1
        return d
    
    def erdosRenyi(n, p):
        """Constructs a $G_{n,p}$ Erdős-Rényi random graph.
        ...
        References
        ----------
        .. [1] P. Erdős and A. Rényi, On Random Graphs, Publ. Math. 6, 290 (1959).
        .. [2] E. N. Gilbert, Random Graphs, Ann. Math. Stat., 30, 1141 (1959).
        """
        A = np.zeros((n,n))
        for i in range(n):
            for j in range(n):
                if np.random.rand() < p:
                    A[i, j] = 1
                    A[j, i] = 1
        return graph(A)
    
    
    def structuralDM(G1, G2, metric='Euclidean'):
        """Computes a graph distance measure defined by comparing the distance of the adjacency matrices 
        between 2 graphs with respect to some metric.
        
        :param G1: A graph object
        :param G2: A graph object
        :param metric: Specify the metric used to compare the graphs in.
        :type metric: Euclidean, Manhattan, Canberra, or Jaccard
        
        :return: A similarity measure between the two graphs.
        References
        ----------
        .. [1] Surana, Amit, Can Chen, and Indika Rajapakse. "Hypergraph dissimilarity measures." arXiv preprint arXiv:2106.08206 (2021).
        """
        print('stub')
        print(G1 + G2)
        
    def spectralDM(G1, G2):
        """Computes a graph distance measure defined by comparing the distance of the adjacency matrices 
        between 2 graphs with respect to some metric.
        
        :param G1: A graph object
        :param G2: A graph object
        
        :return: A similarity measure between the two graphs.
        References
        ----------
        .. [1] Surana, Amit, Can Chen, and Indika Rajapakse. "Hypergraph dissimilarity measures." arXiv preprint arXiv:2106.08206 (2021).
        """
        print('stub')
        print(G1 + G2)

    def featureDM(G1, G2):
        """Computes a graph distance measure defined by comparing the distance of the adjacency matrices 
        between 2 graphs with respect to some metric.
        
        :param G1: A graph object
        :param G2: A graph object
        :param centrality: Specify the centrality measure metric used to compare the graphs.
        :type metric: Eigen, Degree, Betweenness, PageRank
        
        :return: A similarity measure between the two graphs.
        References
        ----------
        .. [1] Surana, Amit, Can Chen, and Indika Rajapakse. "Hypergraph dissimilarity measures." arXiv preprint arXiv:2106.08206 (2021).
        """
        print('stub')
        print(G1 + G2)
