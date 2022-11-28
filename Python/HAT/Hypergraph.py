import numpy as np
import scipy as sp
import scipy.linalg
from itertools import permutations 

from HAT.graph import graph

class Hypergraph:

    def __init__(self, im, es=None, ew=None, nw=None):
        """This is the default constructor for a hypergraph. 
        """
        self.IM = im
        self.edgeWeight = ew
        self.nodeWeight = nw 

        if self.edgeWeight is None:
            self.edgeWeight = np.ones(self.IM.shape[1])
        if self.nodeWeight is None:
            self.nodeWeight = np.ones(self.IM.shape[1])
    
    def dual(self):
        """The dual hypergraph is constructed.
        
        References
        ----------
        .. [1] Yang, Chaoqi, et al. "Hypergraph learning with line expansion." arXiv preprint arXiv:2005.04843 (2020).
        """
        IM = self.IM.T
        return Hypergraph(IM)

    def cliqueGraph(self):
        """The clique expansion graph is constructed.

        References
        ----------
        .. [1] Yang, Chaoqi, et al. "Hypergraph learning with line expansion." arXiv preprint arXiv:2005.04843 (2020).
        """
        edgeOrder = np.sum(self.IM, axis=0)
        M = np.zeros((len(edgeOrder),len(edgeOrder)))
        np.fill_diagonal(M, self.edgeWeight)
        A = self.IM @ M @ self.IM.T
        return graph(A)

    def lineGraph(self):
        """The line graph, which is the clique expansion of the dual graph, is constructed.
        
        References
        ----------
        .. [1] Yang, Chaoqi, et al. "Hypergraph learning with line expansion." arXiv preprint arXiv:2005.04843 (2020).
        """
        D = self.dual()
        return D.cliqueGraph()

    def starGraph(self):
        """The star graph representation is constructed.
        
        References
        ----------
        .. [1] Yang, Chaoqi, et al. "Hypergraph learning with line expansion." arXiv preprint arXiv:2005.04843 (2020).
        """
        
        n = len(self.IM) + len(self.IM[0])
        A = np.zeros((n,n))
        A[len(A) - len(self.IM):len(A),0:len(self.IM[0])] = self.IM
        A[0:len(self.IM[0]),len(A) - len(self.IM):len(A)] = self.IM.T
        return graph(A)
    
    def laplacianMatrix(self, type):
        if type == 'Bolla':
            return self.bolla()
        elif type == 'Rodriguez':
            return self.rodriguez()
        elif type == 'Zhou':
            return self.zhou()
        
    def bolla(self):
        """
        References
        ----------
        .. [1] Bolla, M. (1993). Spectra, euclidean representations and clusterings of hypergraphs. Discrete Mathematics, 117.
               https://www.sciencedirect.com/science/article/pii/0012365X9390322K
        """
        dv = np.sum(self.IM, axis=1)
        Dv = np.zeros((len(dv), len(dv)))
        np.fill_diagonal(Dv, dv)
        de = np.sum(self.IM, axis=0)
        De = np.zeros((len(de), len(de)))
        np.fill_diagonal(De, de)
        DeInv = sp.linalg.inv(De, overwrite_a=True)
        L = Dv - (self.IM @ DeInv @ self.IM.T)
        return L
        
    def rodriguez(self):
        """
        References
        ----------
        .. [1] Rodriguez, J. A. (2002). On the Laplacian eigenvalues and metric parameters of hypergraphs. Linear and Multilinear Algebra, 50(1), 1-14.
               https://www.tandfonline.com/doi/abs/10.1080/03081080290011692
        .. [2] Rodriguez, J. A. (2003). On the Laplacian spectrum and walk-regular hypergraphs. Linear and Multilinear Algebra, 51, 285–297.
               https://www.tandfonline.com/doi/abs/10.1080/0308108031000084374
        """
        A = self.IM @ self.IM.T
        np.fill_diagonal(A, 0)
        L = np.diag(sum(A)) - A
        return L
    
    def zhou(self):
        """
        References
        ----------
        .. [1] Zhou, D., Huang, J., & Schölkopf, B. (2005). Beyond pairwise classification and clustering using hypergraphs. (Equation 3.3)
               https://dennyzhou.github.io/papers/hyper_tech.pdf
        """
        Dvinv = np.diag(1/np.sqrt(self.IM @ self.edgeWeight))
        Deinv = np.diag(1/np.sum(self.IM, axis=0))
        W = np.diag(self.edgeWeight)
        L = np.eye(len(self.IM)) - Dvinv @ self.IM @ W @ Deinv @ self.IM.T @ Dvinv
        return L        
        
    def adjacencyTensor(self):
        """This constructs the adjacency tensor for uniform hypergraphs.
        """
        order = sum(self.IM[:,0])
        denom = np.math.factorial(order - 1)
        modes = len(self.IM) * np.ones(order)
        A = np.zeros(modes.astype(int))
        for e in range(len(self.IM[0])):
            vxc = np.where(self.IM[:, e] == 1)[0]
            vxc = list(permutations(vxc))
            for p in vxc:
                A[p] = 1 / denom
        return A
    
    def degreeTensor(self):
        """This constructs the degree tensor for uniform hypergraphs.
        """
        order = sum(self.IM[:,0])
        modes = len(self.IM) * np.ones(order)
        D = np.zeros(modes.astype(int))
        for vx in range(len(self.IM)):
            D[tuple((np.ones(order) * vx).astype(int))] = sum(self.IM[vx])
        return D
    
    def laplacianTensor(self):
        """This constructs the Laplacian tensor for uniform hypergraphs.
        """
        D = self.degreeTensor()
        A = self.adjacencyTensor()
        L = D - A
        return L
        
    def tensorEntropy(self):
        """Computes hypergraph entropy based on the singular values of the Laplacian tensor. This comes from definition 7 of [1] and
        is implemented with Algorithm 1 from [1].
        
        References
        ----------
        .. [1] C. Chen and I. Rajapakse, Tensor Entropy for Uniform Hypergraphs, IEEE TRANSACTIONS ON NETWORK SCIENCE AND ENGINEERING (2020)
        """
        L = self.laplacienTensor
        L = np.reshape(L, (len(self.IM), len(self.IM)^(sum(W[:,0]) - 1)))
        _, S, _ = sp.linalg.svd(L)
        S /= sum(S)
        return sp.stats.entropy(S)
    
    def avgDistance(self):
        """The average distance is computed between any vertices in the hypergraph based
        on the average distance of the clique expansion.
        """
        g = self.cliqueExpand()
        return g.avgDistance()