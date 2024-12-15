import numpy as np
import scipy as sp

def hosvd(T, M=True, uniform=False, sym=False):
    """ Higher Order Singular Value Decomposition
    
    :param uniform: Indicates if T is a uniform tensor
    :param sym: Indicates if T is a super symmetric tensor
    :param M: Indicates if the factor matrices are required as well as the core tensor
    
    :return: The singular values of the core diagonal tensor and the factor matrices.
    """
    # Auth: Joshua Pickard
    #       jpic@umich.edu
    # Date: Dec 2, 2022
    if uniform and not M:
        return supersymHosvd(T)
    else:
        print('Nonuniform SVD not implemented')
        
def supersymHosvd(T):
    """ Computes the singular values of a uniform, symetric tensor. See Algorithm 1 in [1].
    
    :param T: A uniform, symmetric multidimensional array
    
    :return: The singular values that compose the core tensor of the HOSVD on T.
    
    References
    ----------
    .. [1] C. Chen and I. Rajapakse, Tensor Entropy for Uniform Hypergraphs, IEEE TRANSACTIONS ON NETWORK SCIENCE AND ENGINEERING (2020)
    """
    # Auth: Joshua Pickard
    #       jpic@umich.edu
    # Date: Dec 2, 2022
    d = T.shape[0]
    m = len(T.shape)
    M = np.reshape(T, (d, d**(m-1)))
    return sp.linalg.svd(M)

def HammingSimilarity(A1, A2):
    """Computes the Spectral-S similarity of 2 Adjacency tensors [1].

    :param A1: adjacency tensor 1
    :type A1: *ndarray*
    :param A2: adjacency tensor 2
    :type A2: *ndarray*
    :return: Hamming similarity measure
    :rtype: *float*

    References
    ==========
    .. [1] Amit Surana, Can Chen, and Indika Rajapakse. Hypergraph similarity measures. IEEE Transactions on Network Science and Engineering, pages 1-16, 2022.

    """
    # Auth: Joshua Pickard
    #       jpic@umich.edu
    # Date: Dec 2, 2022
    modes = A1.shape
    order = len(modes)
    s = sum(abs(A1 - A2))
    while len(s.shape) != 1:
        s = sum(s)
    s = sum(s)
    return s / (modes[0]**order - modes[0])

def SpectralHSimilarity(L1, L2):
    """Computes the Spectral-S similarity of 2 Laplacian tensors [1].

    :param L1: Laplacian tensor 1
    :type L1: *ndarray*
    :param L2: Laplacian tensor 2
    :type L2: *ndarray*
    :return: Spectral-S similarity measure
    :rtype: *float*

    References
    ==========
    .. [1] Amit Surana, Can Chen, and Indika Rajapakse. Hypergraph similarity measures. IEEE Transactions on Network Science and Engineering, pages 1-16, 2022.
    """
    # Auth: Joshua Pickard
    #       jpic@umich.edu
    # Date: Dec 2, 2022
    _, S1, _ = supersymHosvd(L1)
    _, S2, _ = supersymHosvd(L2)
    S1 = (1/sum(S1)) * S1
    S2 = (1/sum(S1)) * S2
    S = sum(abs(S1 - S2)**2)/len(S1)
    return S
    
def kronecker_exponentiation(M, x):
    """Kronecker Product Exponential.

    :param M: a matrix
    :type M: *ndarray*
    :param x: power of exponentiation
    :type x: *int*
    :return: Krnoecker Product exponentiation of **M** a total of **x** times
    :rtype: *ndarray*    

    This function performs the Kronecker Product on a matrix :math:`M` a total of 
    :math:`x` times. The Kronecker product is defined for two matrices 
    :math:`A\in\mathbf{R}^{l \\times m}, B\in\mathbf{R}^{m \\times n}` as the matrix

    .. math::
        A \\bigotimes B= \\begin{pmatrix} A_{1,1}B & A_{1,2}B & \dots & A_{1,m}B \\\\ A_{2,1}B & A_{2,2}B & \dots & A_{2,m}B \\\\ \\vdots & \\vdots & \\ddots & \\vdots \\\\ A_{l,1}B & A_{l,2}B & \dots & A_{l,n}B \\end{pmatrix}

    """
    # Auth: Joshua Pickard
    #       jpic@umich.edu
    # Date: Dec 2, 2022

    # Base cases
    if x == 1:
        return M
    elif x % 2 == 0:
        half_kron = kronecker_exponentiation(M, x // 2)
        return np.kron(half_kron, half_kron)
    else:
        return np.kron(M, kronecker_exponentiation(M, x - 1))

    
    