import numpy as np
import scipy as sp
import scipy.linalg
from itertools import permutations
import networkx as nx

import multilinalg as mla
import hypergraph

"""
This file implements tensor representations of hypergraph
"""


def adjTensor(HG):
    """This constructs the adjacency tensor for uniform hypergraphs.
    
    :return: Adjacency Tensor
    :rtype: *ndarray*

    The adjacency tensor :math:`A` of a :math:`k-`order hypergraph :math:`H` is the multi-way, hypergraph analog of the pairwise, graph
    adjacency matrix. It is defined as a :math:`k-` mode tensor ( :math:`k-` dimensional matrix):

    .. math::
        A \in \mathbf{R}^{ \overbrace{n \\times \dots \\times n}^{k \\text{ times}}} \\text{ where }{A}_{j_1\dots j_k} = \\begin{cases} \\frac{1}{(k-1)!} & \\text{if }(j_1,\dots,j_k)\in {E}_h \\\\ 0 & \\text{otherwise} \end{cases},

    as found in equation 8 of [1].

    References
    ==========
    .. [1] C. Chen and I. Rajapakse, Tensor Entropy for Uniform Hypergraphs, IEEE TRANSACTIONS ON NETWORK SCIENCE AND ENGINEERING (2020)
        (Equation 8) https://arxiv.org/pdf/1912.09624.pdf
    """
    # Auth: Joshua Pickard
    #       jpic@umich.edu
    # Date: Nov 30, 2022
    order = int(sum(HG.IM[:,0]))
    denom = np.math.factorial(order - 1)
    modes = len(HG.IM) * np.ones(order)
    A = np.zeros(modes.astype(int))
    for e in range(len(HG.IM[0])):
        vxc = np.where(HG.IM[:, e] == 1)[0]
        vxc = list(permutations(vxc))
        for p in vxc:
            A[p] = 1 / denom
    return A

def degreeTensor(HG):
    """This constructs the degree tensor for uniform hypergraphs.
    
    :return: Degree Tensor
    :rtype: *ndarray*

    The degree tensor :math:`D` is the hypergraph analog of the degree matrix. For a :math:`k-` order hypergraph
    :math:`H=(V,E)` the degree tensor :math:`D` is a diagonal supersymmetric tensor defined

    .. math::
        D \in \mathbf{R}^{ \overbrace{n \\times \dots \\times n}^{k \\text{ times}}} \\text{ where }{D}_{i\dots i} = degree(v_i) \\text{ for all } v_i\in V

    References
    ----------
    .. [1] C. Chen and I. Rajapakse, Tensor Entropy for Uniform Hypergraphs, IEEE TRANSACTIONS ON NETWORK SCIENCE AND ENGINEERING (2020)
        https://arxiv.org/pdf/1912.09624.pdf
    """
    # Auth: Joshua Pickard
    #       jpic@umich.edu
    # Date: Nov 30, 2022
    order = int(sum(HG.IM[:,0]))
    modes = len(HG.IM) * np.ones(order)
    D = np.zeros(modes.astype(int))
    for vx in range(len(HG.IM)):
        D[tuple((np.ones(order) * vx).astype(int))] = sum(HG.IM[vx])
    return D

def laplacianTensor(HG):
    """This constructs the Laplacian tensor for uniform hypergraphs.
    
    :return: Laplcian Tensor
    :rtype: *ndarray*

    The Laplacian tensor is the tensor analog of the Laplacian matrix for graphs, and it is
    defined equivalently. For a hypergraph :math:`H=(V,E)` with an adjacency tensor :math:`A`
    and degree tensor :math:`D`, the Laplacian tensor is

    .. math::
        L = D - A

    References
    ==========        
    .. [1] C. Chen and I. Rajapakse, Tensor Entropy for Uniform Hypergraphs, IEEE TRANSACTIONS ON NETWORK SCIENCE AND ENGINEERING (2020)
        (Equation 9) https://arxiv.org/pdf/1912.09624.pdf
    """
    # Auth: Joshua Pickard
    #       jpic@umich.edu
    # Date: Nov 30, 2022
    D = HG.degreeTensor()
    A = HG.adjTensor()
    L = D - A
    return L
    
def tensorEntropy(HG):
    """Computes hypergraph entropy based on the singular values of the Laplacian tensor.
    
    :return: tensor entropy
    :rtype: *float*

    Uniform hypergraph entropy is defined as the entropy of the higher order singular
    values of the Laplacian matrix [1].
    
    References
    ----------
    .. [1] C. Chen and I. Rajapakse, Tensor Entropy for Uniform Hypergraphs, IEEE TRANSACTIONS
         ON NETWORK SCIENCE AND ENGINEERING (2020) (Definition 7, Algorithm 1) 
         https://arxiv.org/pdf/1912.09624.pdf
    """
    # Auth: Joshua Pickard
    #       jpic@umich.edu
    # Date: Nov 30, 2022
    L = HG.laplacianTensor()
    _, S, _ = mla.hosvd(L, M=False, uniform=True)
    return sp.stats.entropy(S)