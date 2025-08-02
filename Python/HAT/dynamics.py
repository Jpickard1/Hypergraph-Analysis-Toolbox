import numpy as np
import scipy as sp

# from HAT import Hypergraph
from HAT import multilinalg as mla


def detect_hyperedge_dilation(HG)
    G = HG.star_graph
    from networkx.algorithms import bipartite
    matching = nx.algorithms.matching.max_weight_matching(G, maxcardinality=True)
    if len(matching) == HG.nnodes
        dilation_exists = False
    else:
        dilation_exists = True
    return dilation_exists, matching

def walk_breadth_first_search(HG, start_nodes):
    from collections import deque
    if not HG.directed:
        print("HG must be directed")
        return

    accessible_nodes = set(start_nodes)
    visited_edges = set()

    queue = deque()

    # Initial step: add edges whose tails are fully accessible
    for idx, tail in enumerate(HG.edges['tail']):
        if set(tail).issubset(accessible_nodes):
            queue.append(idx)
            visited_edges.add(idx)

    # Breadth-first traversal
    while queue:
        idx = queue.popleft()

        # Add head nodes to accessible set
        head_nodes = HG.edges['head'][idx]
        accessible_nodes.update(head_nodes)

        # Check for new edges that can now be activated
        for jdx, tail in enumerate(HG.edges['tail']):
            if jdx not in visited_edges and set(tail).issubset(accessible_nodes):
                queue.append(jdx)
                visited_edges.add(jdx)

    return accessible_nodes, visited_edges


def ctrbk(HG, inputVxc):
    """Compute the reduced controllability matrix for :math:`k-` uniform hypergraphs.

    :param inputVxc: List of vertices that may be controlled
    :type inputVxc: *ndarray*
    :return: Controllability matrix
    :rtype: *ndarray*

    References
    ==========
      - Chen C, Surana A, Bloch A, Rajapakse I. "Controllability of Hypergraphs."
        IEEE Transactions on Network Science and Engineering, 2021. https://drive.google.com/file/d/12aReE7mE4MVbycZUxUYdtICgrAYlzg8o/view
    """
    # Auth: Joshua Pickard
    #       jpic@umich.edu
    # Date: Nov 30, 2022
    A = HG.adjacency_tensor
    modes = A.shape
    n = modes[0]
    order = len(modes)
    Aflat = np.reshape(A, (n, n**(order-1)))
    ctrbMatrix = b_matrix(HG, inputVxc)
    j = 0
    while j < n and np.linalg.matrix_rank(ctrbMatrix) < n:
        kprod = mla.kronecker_exponentiation(ctrbMatrix, len(modes)-1)
        nextCtrbMatrix = Aflat @ kprod;
        ctrbMatrix = np.concatenate((ctrbMatrix, nextCtrbMatrix), axis=1)
        r = np.linalg.matrix_rank(ctrbMatrix)
        U, _, _ = sp.linalg.svd(ctrbMatrix)
        ctrbMatrix = U[:,0:r]
        j += 1
    return ctrbMatrix
    
def b_matrix(HG, inputVxc):
    """Constructs controllability :math:`B` matrix commonly used in the linear control system
    
    .. math::
        \\frac{dx}{dt} = Ax+Bu

    :param inputVxc: a list of input control nodes
    :type inputVxc: *ndarray*
    :return: control matrix
    :rtype: *ndarray*

    References
    ==========
      - Can Chen, Amit Surana, Anthony M Bloch, and Indika Rajapakse. Controllability of hypergraphs. IEEE Transactions
        on Network Science and Engineering, 8(2):1646–1657, 2021. https://drive.google.com/file/d/12aReE7mE4MVbycZUxUYdtICgrAYlzg8o/view
    """
    # Auth: Joshua Pickard
    #       jpic@umich.edu
    # Date: Nov 30, 2022
    B = np.zeros((HG.nnodes, len(inputVxc)))
    for i in range(len(inputVxc)):
        B[inputVxc[i], i] = 1
    return B

