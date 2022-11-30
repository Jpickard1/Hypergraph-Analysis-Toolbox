import itertools
from itertools import combinations
import numpy as np
import scipy as sp
import networkx as nx

import HAT.multilinalg
import HAT

def directSimilarity(HG1, HG2, measure):
    if measure == 'Hamming':
        return HAT.multilinalg.HammingSimilarity(HG1.laplacianTensor(), HG2.laplacianTensor())
    elif measure == 'Spectral-S':
        return HAT.multilinalg.SpectralHSimilarity(HG1.laplacianTensor(), HG2.laplacianTensor())
    # elif measure == 'Spectral-H': # Not implemented until we get a H-eigenvalue solver
    elif measure == 'centrality':
        C1 = HG1.centrality()[0]
        C2 = HG2.centrality()[0]
        C1 /= np.linalg.norm(C1)
        C2 /= np.linalg.norm(C2)
        m = np.linalg.norm(C1 - C2) / len(C1)
        return m
    
def indirectSimilarity(G1, G2, measure, centralityType=None, eps=10e-3):
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
        s = (1 / len(v1)) * np.linalg.norm(mat1-mat2)
    return s

def multicorrelations(D, order, mtype, idxs=None):
    """
    References
    ----------
    .. [1] Zvi Drezner. Multirelation—a correlation among more than two variables. Computational Statistics & Data Analysis, 19(3):283–292, 1995.
    .. [2] Jianji Wang and Nanning Zheng. Measures of correlation for multiple variables. arXiv preprint arXiv:1401.4827, 2014.
    .. [3] Benjamin M Taylor. A multi-way correlation coefficient. arXiv preprint arXiv:2003.02561, 2020.
    """
    R = np.corrcoef(D.T)
    
    if idxs == None:
        idxs = np.array(list(itertools.combinations(range(4),3)))
    
    M = np.zeros(len(idxs),)
    
    if mtype == 'Taylor':
        taylorCoef = 1/np.sqrt(order)

    for i in range(len(idxs)):
        minor = R[idxs[i,:], :]
        minor = minor[:, idxs[i,:]]
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
    IM = np.zeros((v,e))
    for i in range(e):
        idx = np.random.choice(v, size = k, replace = False)
        IM[idx,i] = 1
    return HAT.Hypergraph(IM)

