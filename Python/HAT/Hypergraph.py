import numpy as np
import scipy as sp
import scipy.linalg
import scipy.stats
from itertools import permutations
import networkx as nx

import HAT.multilinalg as mla
import HAT.draw

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
            self.nodeWeight = np.ones(self.IM.shape[0])
    
    def draw(self):
        HAT.draw.incidencePlot(self)
        
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
        np.fill_diagonal(A,0) # Omit self loops
        return nx.from_numpy_matrix(A)

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
        return nx.from_numpy_matrix(A)
    
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
        
    def adjTensor(self):
        """This constructs the adjacency tensor for uniform hypergraphs.
        """
        order = int(sum(self.IM[:,0]))
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
        order = int(sum(self.IM[:,0]))
        modes = len(self.IM) * np.ones(order)
        D = np.zeros(modes.astype(int))
        for vx in range(len(self.IM)):
            D[tuple((np.ones(order) * vx).astype(int))] = sum(self.IM[vx])
        return D
    
    def laplacianTensor(self):
        """This constructs the Laplacian tensor for uniform hypergraphs.
        """
        D = self.degreeTensor()
        A = self.adjTensor()
        L = D - A
        return L
        
    def tensorEntropy(self):
        """Computes hypergraph entropy based on the singular values of the Laplacian tensor. This comes from definition 7 of [1] and
        is implemented with Algorithm 1 from [1].
        
        References
        ----------
        .. [1] C. Chen and I. Rajapakse, Tensor Entropy for Uniform Hypergraphs, IEEE TRANSACTIONS ON NETWORK SCIENCE AND ENGINEERING (2020)
        """
        
        L = self.laplacianTensor()
        _, S, _ = mla.hosvd(L, M=False, uniform=True)
        #L = np.reshape(L, (len(self.IM), len(self.IM)^(sum(W[:,0]) - 1)))
        #_, S, _ = sp.linalg.svd(L)
        #S /= sum(S)
        return sp.stats.entropy(S)
    
    def matrixEntropy(self, Ltype='Rodriguez'):
        """Computes hypergraph entropy based on the eigenvalues values of the Laplacian matrix.
        """
        L = self.laplacianMatrix(Ltype)
        U, V = np.linalg.eig(L)
        return sp.stats.entropy(U)
    
    def avgDistance(self):
        """The average distance is computed between any vertices in the hypergraph based
        on the average distance of the clique expansion.
        """
        G = self.cliqueGraph()
        return nx.average_shortest_path_length(G)
    
    def ctrbk(self, inputVxc):
        A = self.adjTensor()
        modes = A.shape
        n = modes[0]
        order = len(modes)
        Aflat = np.reshape(A, (n, n**(order-1)))
        # print(Aflat.shape)
        ctrbMatrix = self.BMatrix(inputVxc)
        print(ctrbMatrix.shape)
        j = 0
        while j < n and np.linalg.matrix_rank(ctrbMatrix) < n:
            kprod = mla.kronExponentiation(ctrbMatrix, len(modes)-1)
            # print(kprod.shape)
            nextCtrbMatrix = Aflat @ kprod;
            # print(nextCtrbMatrix.shape)
            ctrbMatrix = np.concatenate((ctrbMatrix, nextCtrbMatrix), axis=1)
            print(ctrbMatrix.shape)
            r = np.linalg.matrix_rank(ctrbMatrix)
            U, _, _ = sp.linalg.svd(ctrbMatrix)
            ctrbMatrix = U[:,0:r]
            print(ctrbMatrix.shape)
            j += 1
        return ctrbMatrix
        
    def BMatrix(self, inputVxc):
        B = np.zeros((len(self.nodeWeight), len(inputVxc)))
        for i in range(len(inputVxc)):
            B[inputVxc[i], i] = 1
        return B
    
    def clusteringCoef(self):
        n, e = self.IM.shape
        order = sum(self.IM[:,0])
        avgClusterCoef = 0
        for v in range(n):
            edges = np.where(self.IM[v,:] > 0)[0]
            neighbors = np.where(np.sum(self.IM[:, edges], axis=1) > 0)[0]
            avgClusterCoef += (len(edges) / sp.special.comb(len(neighbors), order))
        avgClusterCoef /= n
        return avgClusterCoef
                                   
    def centrality(self, tol=1e-4, maxIter=3000, model='LogExp', alpha=10):
        """
        References
        ----------
        .. [1] % Tudisco, F., Higham, D.J. Node and edge nonlinear eigenvector centrality for hypergraphs. Commun Phys 4, 201 (2021). https://doi.org/10.1038/s42005-021-00704-2
        """
        if model=='Linear':
            f = lambda x : x
            g = lambda x : x
            phi = lambda x : x
            psi = lambda x : x
        elif model=='LogExp':
            f = lambda x : x
            g = lambda x : np.sqrt(x)
            phi = lambda x : np.log(x)
            psi = lambda x : np.exp(x)
        elif model=='Max':
            f = lambda x : x
            g = lambda x : x
            phi = lambda x : x**alpha
            psi = lambda x : x**(1/alpha)
        elif isinstance(model, list):
            f = model[0]
            g = model[1]
            phi = model[2]
            psi = model[3]
        else:
            print('Enter a valid centrality model')
            
        B = self.IM
        W = self.edgeWeight
        N = self.nodeWeight
        W = np.diag(W)
        N = np.diag(N)
        n, m = B.shape        
        x0 = np.ones(n,) / n
        y0 = np.ones(m,) / m
        
        for i in range(maxIter):
            # print(np.array(y0).shape)
            # print(B.T)
            # print(N)
            # print(phi(x0))
            # print(np.array(psi(B.T @ N @ phi(x0))).shape)
            u = np.sqrt(np.multiply(np.array(x0),np.array(g(B @ W @ f(y0)))))
            v = np.sqrt(np.multiply(np.array(y0),np.array(psi(B.T @ N @ phi(x0)))))
            x = u / sum(u)
            y = v / sum(v)
            
            check = sum(x - x0) + sum(y - y0)
            if check < tol:
                break
            else:
                x0 = x
                y0 = y
                
        nodeCentrality = x
        edgeCentrality = y
        
        return nodeCentrality, edgeCentrality