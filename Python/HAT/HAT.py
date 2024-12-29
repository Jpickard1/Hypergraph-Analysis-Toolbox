import itertools
from itertools import combinations
import numpy as np
import scipy as sp
import scipy.io
import networkx as nx
import os
import pandas as pd

import multilinalg
# import HAT

"""
HAT.HAT module contains miscilaneous hypergraph methods
"""

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
    # Auth: Joshua Pickard
    #       jpic@umich.edu
    # Date: Dec 2, 2022
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
    # Auth: Joshua Pickard
    #       jpic@umich.edu
    # Date: Dec 2, 2022
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

def multicorrelations(D, order, mtype='Drezner', idxs=None, v=False, vfreq=1000):
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
    :param v: verbose, defaults to False
    :type v: bool, optional
    :param vfreq: frequency to display output in verbose mode
    :type vfrea: int, optional
    :return: A vector of the multicorrelation scores computed and a vector of the column indices of
        ``D`` used to compute each multicorrelation.
    :rtype: *(ndarray, ndarray)*

    References
    ----------
    .. [1] Zvi Drezner. Multirelation—a correlation among more than two variables. Computational Statistics & Data Analysis, 19(3):283–292, 1995.
    .. [2] Jianji Wang and Nanning Zheng. Measures of correlation for multiple variables. arXiv preprint arXiv:1401.4827, 2014.
    .. [3] Benjamin M Taylor. A multi-way correlation coefficient. arXiv preprint arXiv:2003.02561, 2020.
    """
    # Auth: Joshua Pickard
    #       jpic@umich.edu
    # Date: Dec 2, 2022
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

        if v and i % vfreq == 0:
            print(f"{i} / {len(idxs)} mcorrs complete")

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
    # Auth: Joshua Pickard
    #       jpic@umich.edu
    # Date: Dec 2, 2022
    IM = np.zeros((v,e))
    for i in range(e):
        idx = np.random.choice(v, size = k, replace = False)
        IM[idx,i] = 1
    return HAT.Hypergraph(IM)

def load(dataset='Karate'):
    """This function loads built-in datasets. Currently only one dataset is available and we are working to expand this.

    :param dataset: sets which dataset to load in, defaults to 'Karate'
    :type dataset: str, optional
    :return: incidence matrix or graph object
    :rtype: *ndarray* or *nx.Graph*
    """
    # Auth: Joshua Pickard
    #       jpic@umich.edu
    # Date: Dec 2, 2022

    current_path = os.path.dirname(os.path.realpath(__file__))
    current_path += '/data/'
    if dataset == 'Karate':
        return nx.karate_club_graph()
    elif dataset == 'ArnetMiner Citation':
        mat = sp.io.loadmat(current_path + 'aminer_cocitation.mat')
        S = mat['S']
        return S
    elif dataset == 'ArnetMiner Reference':
        mat = sp.io.loadmat(current_path + 'aminer_coreference.mat')
        S = mat['S']
        return S
    elif dataset == 'Citeseer Citation':
        mat = sp.io.loadmat(current_path + 'citeseer_cocitation.mat')
        S = mat['S']
        return S
    elif dataset == 'Cora Citation':
        mat = sp.io.loadmat(current_path + 'cora_coreference.mat')
        S = mat['S']
        return S
    elif dataset == 'Cora Citation':
        mat = sp.io.loadmat(current_path + 'cora_coreference.mat')
        S = mat['S']
        return S
    elif dataset == 'DBLP':
        mat = sp.io.loadmat(current_path + 'dblp.mat')
        S = mat['S']
        return S

def hyperedges2IM(edgeSet):
    """This function constructs an incidence matrix from an edge set.

    :param edgeSet: a :math:`e \\times k` matrix where each row contains :math:`k` integers that are contained within the same hyperedge
    :type edgeSet: *ndarray*
    :return: a :math:`n \times e` incidence matrix where each row of the edge set corresponds to a column of the incidence matrix. :math:`n` is the number of nodes contained in the edgeset.
    :rtype: *ndarray*
    """
    # Auth: Joshua Pickard
    #       jpic@umich.edu
    # Date: Dec 2, 2022
    n = np.max(edgeSet)
    e = len(edgeSet)
    IM = np.zeros((n+1,e))
    for e in range(e):
        IM[edgeSet[e,:],e] = 1
    return IM
    
def hyperedgeHomophily(H, HG=None, G=None, method='CN'):
    """This function computes the hyperedge homophily score according to the below methods. The homophily score is the average score based on
    structural similarity of the vertices in hypredge `H` in the clique expanded graph `G`. This function is an interface from `HAT` to `networkx`
    link prediction algorithms.

    :param G: a pairwise hypergraph expansion
    :type G: `networkx.Graph`
    :param H: hyperedge containing individual vertices within the edge
    :type H: `ndarray`
    :param method: specifies which structural similarity method to use. This defaults to `CN` common neighbors.
    """
    # Auth: Joshua Pickard
    #       jpic@umich.edu
    # Date: Dec 6, 2022
    pairwise = list(itertools.combinations(H, 2))

    # Compute pairwise scores with networkx
    if method == 'CN':
        pairwiseScores = nx.common_neighbor_centrality(G, pairwise)
    elif method == 'RA':
        pairwiseScores = nx.resource_allocation_index(G, pairwise)
    elif method == 'JC':
        pairwiseScores = nx.jaccard_coefficient(G, pairwise)
    elif method == 'AA':
        pairwiseScores = nx.adamic_adar_index(G, pairwise)
    elif method == 'PA':
        pairwiseScores = nx.preferential_attachment(G, pairwise)

    # Compute average pairwise score
    pairwiseScores = pairwiseScores[:, 2]
    hyperedgeHomophily = sum(pairwiseScores)/len(pairwiseScores)
    return hyperedgeHomophily

def edgeRemoval(HG, p, method='Random'):
    """This function randomly removes edges from a hypergraph. In [1], four primary reasons are given for data missing in pairwise networks:
        1. random edge removal
        2. right censoring
        3. snowball effect
        4. cold-ends
    This method removes edes from hypergraphs according to the multi-way analogue of these.
    
    References
    ----------
    .. [1] Yan, Bowen, and Steve Gregory. "Finding missing edges and communities in incomplete networks." Journal of Physics A: Mathematical and Theoretical 
        44.49 (2011): 495102.
    .. [2] Zhu, Yu-Xiao, et al. "Uncovering missing links with cold ends." Physica A: Statistical Mechanics and its Applications 391.22 (2012): 5769-5778.
    """
    # Auth: Joshua Pickard
    #       jpic@umich.edu
    # Date: Dec 9, 2022
    IM = HG.IM
    n, e = IM.shape
    
    if method == 'Random':
        # Randome edge removal
        known, unknown = randomRemoval(HG, p)
    elif method == 'RC':
        # Right censoring
        known, unknown = rightCensorRemoval(HG, p)
    elif method == 'SB':
        # Snowball Effect
        known, unknown = snowBallRemoval(HG, p)
    elif method == 'CE':
        # Cold Ends
        known, unknown = coldEndsRemoval(HG, p)
    else:
        print('Enter a valid edge removal method')
        return
    
    # Bradcast from 3d to 2d
    a, _, c = unknown.shape
    unknown = np.reshape(unknown, (a, c))
    a, _, c = known.shape
    known = np.reshape(known, (a, c))

    # Return hypergraph objects
    K = HAT.Hypergraph(known)
    U = HAT.Hypergraph(unknown)

    return K, U

def randomRemoval(HG, p):
    # Auth: Joshua Pickard
    #       jpic@umich.edu
    # Date: Dec 9, 2022
    IM = HG.IM
    n, e = IM.shape
    knownIdxs = np.random.choice([0, 1], size=(e,), p=[p, 1-p])
    known = IM[:, np.where(knownIdxs == 1)]
    unknown = IM[:, np.where(knownIdxs == 0)]
    return known, unknown

def rightCensorRemoval(HG, p):
    # Auth: Joshua Pickard
    #       jpic@umich.edu
    # Date: Dec 9, 2022
    IM = HG.IM.copy()
    n, e = IM.shape
    
    # Determine number of known edges in remaining graph
    numknownEdges = sum(np.random.choice([0, 1], size=(e,), p=[p, 1-p]))

    # Vertex degree
    vxDegree = np.sum(IM, axis=1)
    
    # Iteratively remove edges
    removedEdges = 0
    knownIdxs = np.ones(e,)
    while sum(knownIdxs) > numknownEdges:
        # Select vertex with maximum degree
        vx = np.argmax(vxDegree)
        # Select edges vx participates in
        vxEdges = np.where(IM[vx,:] != 0)[0]
        # Select single edge to remove
        removeEdge = np.random.choice(vxEdges)
        # Remove edge from incidence matrix
        IM[:, removeEdge] = 0
        # Remove it from list of known edges
        knownIdxs[removeEdge] = 0
        # Decrease degree of vx
        vxDegree[vx] -= 1
    print(IM)
    print(HG.IM)
    known = HG.IM[:, np.where(knownIdxs == 1)]
    unknown = HG.IM[:, np.where(knownIdxs == 0)]
    return known, unknown

def coldEndsRemoval(HG, p):
    # Auth: Joshua Pickard
    #       jpic@umich.edu
    # Date: Dec 9, 2022
    IM = HG.IM.copy()
    n, e = IM.shape
    
    # Determine number of known edges in remaining graph
    numknownEdges = sum(np.random.choice([0, 1], size=(e,), p=[p, 1-p]))

    # Vertex degree
    vxDegree = np.sum(IM, axis=1)
    
    # Iteratively remove edges
    removedEdges = 0
    
    knownIdxs = np.ones(e,)
    while sum(knownIdxs) > numknownEdges:
        # Select vertex with maximum degree
        vx = np.argmin(vxDegree)
        if vxDegree[vx] == 1:
            # Does not remove vxc with degree one. The degree value is set
            # to max and the process continues
            vxDegree[vx] = max(vxDegree)
            continue
        # Select edges vx participates in
        vxEdges = np.where(IM[vx,:] != 0)[0]
        # Select single edge to remove
        removeEdge = np.random.choice(vxEdges)
        # Remove edge from incidence matrix
        IM[:, removeEdge] = 0
        # Remove it from list of known edges
        knownIdxs[removeEdge] = 0
        # Decrease degree of vx
        vxDegree[vx] -= 1
    
    known = HG.IM[:, np.where(knownIdxs == 1)]
    unknown = HG.IM[:, np.where(knownIdxs == 0)]
    return known, unknown

def snowBallRemoval(HG, p):
    # Auth: Joshua Pickard
    #       jpic@umich.edu
    # Date: Dec 9, 2022
    n, e = HG.IM.shape
    source = np.random.choice(n)
    # Clique expand HG
    C = HG.cliqueGraph()
    # Perform BFS on C. BFSedgeList is an ordered list of tuples containing each edge discovered in BFS
    BFSedgeList = list(nx.bfs_tree(C, 5).edges())
    # List of tuples to list preserving order nodes are visited
    orderedVxc = [item for e in BFSedgeList for item in e]
    _, idx = np.unique(orderedVxc, return_index=True)
    # Ordered list vertices are visited
    vxOrder = np.array(orderedVxc)[np.sort(idx)]

    # Set which vertices will remain known in the hypergraph
    numKnownVxc = sum(np.random.choice([0, 1], size=(n,), p=[p, 1-p]))    
    knownVxc = vxOrder[0:numKnownVxc]
    
    # Only include edges where every vertex in the edge is known
    knownIdxs = np.ones(e,)
    for edge in range(e):
        edgeVxc = np.where(HG.IM[:, edge] != 0)
        numRecognizedNodes = np.intersect1d(edgeVxc, knownVxc)
        if len(numRecognizedNodes) != len(edgeVxc):
            knownIdxs[edge] = 0

    known = HG.IM[:, np.where(knownIdxs == 1)]
    unknown = HG.IM[:, np.where(knownIdxs == 0)]
    return known, unknown
    
    