import numpy as np
import scipy as sp
# from HAT import Hypergraph

"""
This file implements hypergraph laplacian methods.
"""

def laplacian_matrix(HG, laplacian_type='Bolla'):
    """This function returns a version of the higher order Laplacian matrix of the hypergraph.

    :param type: Indicates which version of the Laplacin matrix to return. It defaults to ``Bolla`` [1], but ``Rodriguez`` [2,3] and ``Zhou`` [4] are valid arguments as well.
    :type type: str, optional
    :return: Laplacian matrix
    :rtype: *ndarray*

    Several version of the hypergraph Laplacian are defined in [1-4]. These aim to capture
    the higher order structure as a matrix. This function serves as a wrapper to call functions
    that generate different specific Laplacians (See ``bolla_laplacian()``, ``rodriguez_laplacian()``,
    and ``zhou_laplacian()``).

    References
    ----------
    .. [1] Bolla, M. (1993). Spectra, euclidean representations and clusterings of hypergraphs. Discrete Mathematics, 117. 
        https://www.sciencedirect.com/science/article/pii/0012365X9390322K
    .. [2] Rodriguez, J. A. (2002). On the Laplacian eigenvalues and metric parameters of hypergraphs. Linear and Multilinear Algebra, 50(1), 1-14.
        https://www.tandfonline.com/doi/abs/10.1080/03081080290011692
    .. [3] Rodriguez, J. A. (2003). On the Laplacian spectrum and walk-regular hypergraphs. Linear and Multilinear Algebra, 51, 285–297.
        https://www.tandfonline.com/doi/abs/10.1080/0308108031000084374
    .. [4] Zhou, D., Huang, J., & Schölkopf, B. (2005). Beyond pairwise classification and clustering using hypergraphs. (Equation 3.3)
        https://dennyzhou.github.io/papers/hyper_tech.pdf
    """
    # Auth: Joshua Pickard
    #       jpic@umich.edu
    # Date: Nov 30, 2022
    if laplacian_type.lower() == 'bolla':
        return bolla_laplacian(HG)
    elif laplacian_type.lower() == 'rodriguez':
        return rodriguez_laplacian(HG)
    elif laplacian_type.lower() == 'zhou':
        return zhou_laplacian(HG)
    
def bolla_laplacian(HG):
    """This function constructs the hypergraph Laplacian according to [1].

    :return: Bolla Laplacian matrix
    :rtype: *ndarray*

    References
    ----------
    .. [1] Bolla, M. (1993). Spectra, euclidean representations and clusterings of hypergraphs. Discrete Mathematics, 117.
           https://www.sciencedirect.com/science/article/pii/0012365X9390322K
    """
    # Auth: Joshua Pickard
    #       jpic@umich.edu
    # Date: Nov 30, 2022
    dv = np.sum(HG.incidence_matrix, axis=1)
    Dv = np.zeros((len(dv), len(dv)))
    np.fill_diagonal(Dv, dv)
    de = np.sum(HG.incidence_matrix, axis=0)
    De = np.zeros((len(de), len(de)))
    np.fill_diagonal(De, de)
    DeInv = sp.linalg.inv(De, overwrite_a=True)
    L = Dv - (HG.incidence_matrix @ DeInv @ HG.incidence_matrix.T)
    return L
    
def rodriguez_laplacian(HG):
    """This function constructs the hypergraph Laplacian according to [1, 2].

    :return: Rodriguez Laplacian matrix
    :rtype: *ndarray*

    References
    ----------
    .. [1] Rodriguez, J. A. (2002). On the Laplacian eigenvalues and metric parameters of hypergraphs. Linear and Multilinear Algebra, 50(1), 1-14.
           https://www.tandfonline.com/doi/abs/10.1080/03081080290011692
    .. [2] Rodriguez, J. A. (2003). On the Laplacian spectrum and walk-regular hypergraphs. Linear and Multilinear Algebra, 51, 285–297.
           https://www.tandfonline.com/doi/abs/10.1080/0308108031000084374
    """
    # Auth: Joshua Pickard
    #       jpic@umich.edu
    # Date: Nov 30, 2022
    A = HG.incidence_matrix @ HG.incidence_matrix.T
    np.fill_diagonal(A, 0)
    L = np.diag(sum(A)) - A
    return L

def zhou_laplacian(HG):
    """This function constructs the hypergraph Laplacian according to [1].

    :return: Zhou Laplacian matrix
    :rtype: *ndarray*

    References
    ----------
    .. [1] Zhou, D., Huang, J., & Schölkopf, B. (2005). Beyond pairwise classification and clustering using hypergraphs. (Equation 3.3)
           https://dennyzhou.github.io/papers/hyper_tech.pdf
    """
    # Auth: Joshua Pickard
    #       jpic@umich.edu
    # Date: Nov 30, 2022
    Dvinv = np.diag(1/np.sqrt(HG.incidence_matrix @ HG.edge_weights))
    Deinv = np.diag(1/np.sum(HG.incidence_matrix, axis=0))
    W = np.diag(HG.edge_weights)
    L = np.eye(len(HG.incidence_matrix)) - Dvinv @ HG.incidence_matrix @ W @ Deinv @ HG.incidence_matrix.T @ Dvinv
    return L

