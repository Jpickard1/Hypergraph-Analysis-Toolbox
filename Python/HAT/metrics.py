"""
Hypergraph metrics:

1. matrix entropy
2. average distance
3. clustering coefficient
4. nonlinear_eigenvector_centrality
"""

import numpy as np
import networkx as nx
import scipy as sp

def matrix_entropy(HG, laplacian_type='Rodriguez'):
    """Computes hypergraph entropy based on the eigenvalues values of the Laplacian matrix.

    :param laplacian_type: Type of hypergraph Laplacian matrix. This defaults to 'Rodriguez' and other
        choices inclue ``Bolla`` and ``Zhou`` (See: ``laplacianMatrix()``).
    :type laplacian_type: *str, optional*
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
    L = HG.laplacianMatrix(laplacian_type)
    U, V = np.linalg.eig(L)
    return sp.stats.entropy(U)

def avgerage_distance(HG):
    """Computes the average pairwise distance between any 2 vertices in the hypergraph.

    :return: avgDist
    :rtype: *float*

    The hypergraph is clique expanded to a graph object, and the average shortest path on
    the clique expanded graph is returned.
    """
    # Auth: Joshua Pickard
    #       jpic@umich.edu
    # Date: Nov 30, 2022
    G = HG.cliqueGraph()
    avgDist = nx.average_shortest_path_length(G)
    return avgDist

def clustering_coefficient(HG):
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
    n, e = HG.nnodes, HG.nedges
    order = sum(HG.incidence_matrix[:,0])
    avgClusterCoef = 0
    for v in range(n):
        edges = np.where(HG.IM[v,:] > 0)[0]
        neighbors = np.where(np.sum(HG.IM[:, edges], axis=1) > 0)[0]
        avgClusterCoef += (len(edges) / sp.special.comb(len(neighbors), order))
    avgClusterCoef /= n
    return avgClusterCoef
                               
def nonlinear_eigenvector_centrality(HG, tol=1e-4, maxIter=3000, model='LogExp', alpha=10):
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
        
    B = HG.incidence_matrix
    W = HG.edge_weights
    N = HG.node_weights
    W = np.diag(W)
    N = np.diag(N)
    n, m = B.shape
    x0 = np.ones(n,) / n
    y0 = np.ones(m,) / m
    
    for _ in range(maxIter):
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
            
    vertex_centrality = x
    edge_centrality = y
    
    return vertex_centrality, edge_centrality

