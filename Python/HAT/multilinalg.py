import numpy as np
import scipy as sp

def hosvd(T, M=True, uniform=False, sym=False):
    """ Higher Order Singular Value Decomposition
    
    :param uniform: Indicates if T is a uniform tensor
    :param sym: Indicates if T is a super symmetric tensor
    :param M: Indicates if the factor matrices are required as well as the core tensor
    
    :return: The singular values of the core diagonal tensor and the factor matrices.
    """
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
    d = T.shape[0]
    m = len(T.shape)
    M = np.reshape(T, (d, d**(m-1)))
    return sp.linalg.svd(M)

def multirelation(D,order=3,type='Drezner'):
    R = np.corrcoef(D)
    cbns = itertools.combinations(range(len(D[0])),order)
    for i in range(len(cbns)):
        r = R[cbns[i], cbns]
    if type == 'Drezner':
        M = drezner(D, order)
    elif type == 'Taylor':
        M = taylor(D, order)
    else:
        M = wangZheng(D, order)
    return M

# def drezner(D, order):

    