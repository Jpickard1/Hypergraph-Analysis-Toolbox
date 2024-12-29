import numpy as np
import scipy as sp

# from HAT import Hypergraph
from HAT import multilinalg as mla


"""
This contains miscilaneous methods related to hypergraph dynamics, controllability, and observability.
"""

def ctrbk(HG, inputVxc):
    """Compute the reduced controllability matrix for :math:`k-` uniform hypergraphs.

    :param inputVxc: List of vertices that may be controlled
    :type inputVxc: *ndarray*
    :return: Controllability matrix
    :rtype: *ndarray*

    References
    ==========
    .. [1] Chen C, Surana A, Bloch A, Rajapakse I. "Controllability of Hypergraphs."
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
    .. [1] Can Chen, Amit Surana, Anthony M Bloch, and Indika Rajapakse. Controllability of hypergraphs. IEEE Transactions
        on Network Science and Engineering, 8(2):1646–1657, 2021. https://drive.google.com/file/d/12aReE7mE4MVbycZUxUYdtICgrAYlzg8o/view
    """
    # Auth: Joshua Pickard
    #       jpic@umich.edu
    # Date: Nov 30, 2022
    B = np.zeros((HG.nnodes, len(inputVxc)))
    for i in range(len(inputVxc)):
        B[inputVxc[i], i] = 1
    return B