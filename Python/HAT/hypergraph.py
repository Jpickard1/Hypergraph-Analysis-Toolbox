import numpy as np
import scipy as sp

from HAT.graph import graph

class hypergraph:
    """This class represents multiway hypergraph structures.
    """

    def __init__(self):
        """This is the default constructor for a hypergraph.
        """
        self.W = np.zeros((0,0))
        self.N = 0
        self.E = 0
        
    def __init__(self, w):
        """This constructor initializes a hypergraph based on an incidence matrix.
        """
        self.W = w
        self.N = len(self.W)
        self.E = len(self.W[0])
    
    def avgDistance(self):
        """The average distance is computed between any vertices in the hypergraph based
        on the average distance of the clique expansion.
        """
        g = self.cliqueExpand()
        return g.avgDistance()
    
    def dualGraph(self):
        """The dual hypergraph is constructed.
        """
        W = self.W.T
        return hypergraph(W)

    def cliqueExpand(self):
        """The clique expansion graph is constructed.
        """
        A = np.zeros((len(self.W), len(self.W)))
        for e in range(len(self.W[0])):
            vxc = np.where(self.W[:, e] == 1)[0]
            for i in range(len(vxc)):
                for j in range(i+1, len(vxc)):
                    A[vxc[i], vxc[j]] = 1
        A = A + A.T
        return graph(A)

    def lineGraph(self):
        """The line graph, which is the clique expansion of the dual graph is constructed.
        """
        return self.cliqueGraph(self.dualGraph())

    def starGraph(self):
        """The star graph representation is constructed.
        """
        A = np.zeros((len(self.W) + len(self.W[0]), len(self.W) + len(self.W[0])))
        for e in range(len(self.W[0])):
            vxc = np.where(self.W[:, e] == 1)[0]
            A[vxc, len(self.W) + e] = 1
        A = A + A.T
        return graph(A)
        
    def adjacencyTensor(self):
        """This constructs the adjacency tensor for uniform hypergraphs.
        """
        order = sum(self.W[:,0])
        modes = len(self.W) * np.ones(order)
        A = np.zeros(modes.astype(int))
        for e in range(len(W[0])):
            vxc = np.where(W[:, e] == 1)[0]
            vxc = list(permutations(vxc))
            for p in vxc:
                A[p] = 1
        return A
    
    def degreeTensor(self):
        """This constructs the degree tensor for uniform hypergraphs.
        """
        order = sum(self.W[:,0])
        modes = len(self.W) * np.ones(order)
        D = np.zeros(modes.astype(int))
        for vx in range(len(W)):
            D[(np.ones(order) * vx).astype(int)] = sum(W[vx])
        return D
    
    def laplacianTensor(self):
        """This constructs the Laplacian tensor for uniform hypergraphs.
        """
        D = self.degreeTensor()
        A = self.adjacencyTensor()
        L = D - A
        return L
    