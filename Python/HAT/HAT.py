import itertools
from itertools import combinations
import numpy as np
import scipy as sp
import networkx as nx

import HAT.multilinalg
import HAT

def directSimilarity(HG1, HG2, measure='Hamming'):
    """This function computes the direct similarity between two uniform hypergraphs.

    :param HG1: Hypergraph 1
    :type HG1: *Hypergraph*
    :param HG2: Hypergraph 2
    :type HG2: *Hypergraph*
    :param measure: This sepcifies which similarity measure to apply. It defaults to
        ``Hamming``, and ``Spectral-S`` and ``Centrality`` are available as other options
        as well.
    :type measure: str, optional
    :return: Hypergraph similarity
    :rtype: *float*

    References
    ----------
    .. [1] Amit Surana, Can Chen, and Indika Rajapakse. Hypergraph similarity measures. IEEE Transactions on Network Science and Engineering, pages 1-16, 2022.
    """
    if measure == 'Hamming':
        return HAT.multilinalg.HammingSimilarity(HG1.laplacianTensor(), HG2.laplacianTensor())
    elif measure == 'Spectral-S':
        return HAT.multilinalg.SpectralHSimilarity(HG1.laplacianTensor(), HG2.laplacianTensor())
    # elif measure == 'Spectral-H': # Not implemented until we get a H-eigenvalue solver
    elif measure == 'Centrality':
        C1 = HG1.centrality()[0]
        C2 = HG2.centrality()[0]
        C1 /= np.linalg.norm(C1)
        C2 /= np.linalg.norm(C2)
        m = np.linalg.norm(C1 - C2) / len(C1)
        return m
    
def indirectSimilarity(G1, G2, measure='Hamming', eps=10e-3):
    """This function computes the indirect similarity between two hypergraphs.

    :param G1: Hypergraph 1 expansion
    :type G1: *nx.Graph* or *ndarray*
    :param G2: Hypergraph 2 expansion
    :type G2: *nx.Graph* or *ndarray*
    :param measure: This specifies which similarity measure to apply. It defaults to ``Hamming`` , and
        ``Jaccard`` , ``deltaCon`` , ``Spectral`` , and ``Centrality`` are provided as well. When ``Centrality``
        is used as the similarity measure, ``G1`` and ``G2`` should *ndarray* s of centrality values; Otherwise
        ``G1`` and ``G2`` are *nx.Graph*s or *ndarray** s as adjacency matrices.
    :type measure: *str*, optional
    :param eps: a hyperparameter required for deltaCon similarity, defaults to 10e-3
    :type eps: *float*, optional
    :return: similarity measure
    :rtype: *float*

    References
    ==========
    .. [1] Amit Surana, Can Chen, and Indika Rajapakse. Hypergraph similarity measures. IEEE Transactions on Network Science and Engineering, pages 1-16, 2022.
    """
    if isinstance(G1, nx.classes.graph.Graph):
        M1 = nx.adjacency_matrix(G1).todense()
        M1 = np.array(M1)
    else:
        M1 = G1
    if isinstance(G2, nx.classes.graph.Graph):
        M2 = nx.adjacency_matrix(G2).todense()
        M2 = np.array(M2)
    else:
        M2 = G2
    if measure == 'Hamming':
        n = len(M1)
        s = np.linalg.norm(M1 - M2) / (n*(n-1))
    elif measure == 'Jaccard':
        M1 = np.matrix.flatten(M1)
        M2 = np.matrix.flatten(M2)
        M = np.array([M1, M2])
        Mmin = np.min(M, axis=0)
        Mmax = np.max(M, axis=0)
        s = 1 - sum(Mmin)/sum(Mmax)
    elif measure == 'deltaCon':
        D1 = np.diag(sum(M1))
        D2 = np.diag(sum(M2))
        I = np.eye(len(D1))
        S1 = sp.linalg.inv(I - (eps**2) * D1 - (eps*M1))
        S2 = sp.linalg.inv(I - (eps**2) * D2 - (eps*M2))
        M = np.square(np.sqrt(S1) - np.sqrt(S2))
        s = (1 / (len(M1)**2)) * np.sqrt(np.sum(np.sum(M)))
    elif measure == 'Spectral':
        v1, _ = np.linalg.eigh(M1)
        v2, _ = np.linalg.eigh(M2)
        s = (1 / len(v1)) * np.linalg.norm(v1-v2)
    elif measure == 'Centrality':
        # In this case mat1 and mat2 are centrality vectors
        s = (1 / len(v1)) * np.linalg.norm(M1-M2)
    return s

def multicorrelations(D, order, mtype='Drezner', idxs=None):
    """This function computes the multicorrelation among pairwise or 2D data.

    :param D: 2D or pairwise data
    :type D: *ndarray*
    :param order: order of the multi-way interactions
    :type order: *int*
    :param mtype: This specifies which multicorrelation measure to use. It defaults to
        ``Drezner`` [1], but ``Wang`` [2] and ``Taylor`` [3] are options as well.
    :type mtype: *str*
    :param idxs: specify which indices of ``D`` to compute multicorrelations of. The default is ``None``, in which case
        all combinations of ``order`` indices are computed.
    :type idxs: *ndarray*, optional
    :return: A vector of the multicorrelation scores computed and a vector of the column indices of
        ``D`` used to compute each multicorrelation.
    :rtype: *(ndarray, ndarray)*

    References
    ----------
    .. [1] Zvi Drezner. Multirelation—a correlation among more than two variables. Computational Statistics & Data Analysis, 19(3):283–292, 1995.
    .. [2] Jianji Wang and Nanning Zheng. Measures of correlation for multiple variables. arXiv preprint arXiv:1401.4827, 2014.
    .. [3] Benjamin M Taylor. A multi-way correlation coefficient. arXiv preprint arXiv:2003.02561, 2020.
    """

    R = np.corrcoef(D.T)
    
    if idxs == None:
        idxs = np.array(list(itertools.combinations(range(len(R)), order)))
    
    M = np.zeros(len(idxs),)
    
    if mtype == 'Taylor':
        taylorCoef = 1/np.sqrt(order)

    for i in range(len(idxs)):
        minor = R[np.ix_(idxs[i,:], idxs[i,:])]
        if mtype == 'Drezner':
            w, _ = np.linalg.eigh(minor)
            M[i] = 1 - w[0]
        elif mtype == 'Wang':
            M[i] = pow((1 - np.linalg.det(minor)), 0.5)
        elif mtype == 'Taylor':
            w, _ = np.linalg.eigh(minor)
            M[i] = taylorCoef * np.std(w)

    return M, idxs

def uniformErdosRenyi(v, e, k):
    """This function generates a uniform, random hypergraph.

    :param v: number of vertices
    :type v: *int*
    :param e: number of edges
    :type e: *int*
    :param k: order of hypergraph
    :type k: *int*
    :return: Hypergraph
    :rtype: *Hypergraph*
    """
    IM = np.zeros((v,e))
    for i in range(e):
        idx = np.random.choice(v, size = k, replace = False)
        IM[idx,i] = 1
    return HAT.Hypergraph(IM)

def load(dataset):
    """This function loads built-in datasets.
    """
    if dataset == 'Karate':
        return nx.karate_club_graph()

def hyperedges2IM(edgeSet):
    n = np.max(edgeSet)
    e = len(edgeSet)
    IM = np.zeros((n+1,e))
    for e in range(n):
        IM[edgeSet[e,:],:] = 1
    return IM
    