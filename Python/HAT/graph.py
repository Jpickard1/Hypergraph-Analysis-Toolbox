import numpy as np
import scipy as sp

class graph:
    """This class represents pairwise graph structures.
    """
    def __init__(self):
        """Constructs a default graph.
        """
        self.A = np.zeros((0,0))
        self.N = 0
        self.E = 0
        
    def __init__(self, A):
        """Constructs a graph from an adjacency matrix.
        """
        self.A = A
        self.N = len(A)
    
    @property
    def degree(self):
        """The degree matrix of a graph.
        """
        return np.diag(sum(self.A))
        
    @property
    def laplacian(self):
        """The Laplacian matrix of a graph.
        """
        return self.degree - self.A

    @property
    def clusteringCoef(self):
        """The clustering coefficient of the graph.
        """
        gammas = 0
        for vx in range(len(self.A)):
            gammas += self.clusteringCoef(vx)
        return gammas / len(self.A)
        
    def clusteringCoef(self, vx):
        """Computes the clustering coefficient of a single vertex.
        """
        neighbors = np.where(self.A[vx,:] == 1)[0]
        if vx not in neighborhood:
            neighbors = np.append(neighbors, vx)
        neighborhood = self.A[neighbors, :]
        neighborhood = neighborhood[:, neighbors]
        realEdges = sum(neighborhood) / 2
        possibleEdges = sp.special.binom(len(neighbors), 2)
        return realEdges / possibleEdges        

    def pairwiseDistance(self, vxi, vxj):
        """Computes the pairwise distance between vertices i and j.
        """
        d = 1
        N = self.A
        print(N)
        while N[vxi,vxj] == 0:
            print(N)
            N = np.matmul(N, self.A)
            d += 1
        return d
    
    def erdosRenyi(n, p):
        """Constructs a $G_{n,p}$ Erdős-Rényi random graph.
        ...
        References
        ----------
        .. [1] P. Erdős and A. Rényi, On Random Graphs, Publ. Math. 6, 290 (1959).
        .. [2] E. N. Gilbert, Random Graphs, Ann. Math. Stat., 30, 1141 (1959).
        """
        A = np.zeros((n,n))
        for i in range(n):
            for j in range(n):
                if np.random.rand() < p:
                    A[i, j] = 1
                    A[j, i] = 1
        return graph(A)