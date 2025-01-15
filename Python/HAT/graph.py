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
      - Amit Surana, Can Chen, and Indika Rajapakse. Hypergraph similarity measures. IEEE Transactions on Network Science and Engineering, pages 1-16, 2022.
      - Yang, Chaoqi, et al. "Hypergraph learning with line expansion." arXiv preprint arXiv:2005.04843 (2020).
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
      - Yang, Chaoqi, et al. "Hypergraph learning with line expansion." arXiv preprint arXiv:2005.04843 (2020).
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
      - Yang, Chaoqi, et al. "Hypergraph learning with line expansion." arXiv preprint arXiv:2005.04843 (2020).
    """
    # Auth: Joshua Pickard
    #       jpic@umich.edu
    # Date: Nov 30, 2022
    n = len(HG.incidence_matrix) + len(HG.incidence_matrix[0])
    A = np.zeros((n,n))
    A[len(A) - len(HG.incidence_matrix):len(A),0:len(HG.incidence_matrix[0])] = HG.incidence_matrix
    A[0:len(HG.incidence_matrix[0]),len(A) - len(HG.incidence_matrix):len(A)] = HG.incidence_matrix.T
    return nx.from_numpy_array(A)


