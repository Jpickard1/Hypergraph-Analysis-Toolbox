import numpy as np
import scipy as sp

from HAT.graph import graph as G

class hypergraph:

    def __init__(self):
        self.W = np.zeros((0,0))
        self.N = 0
        self.E = 0
        
    def __init__(self, w):
        self.W = w
        self.N = len(self.W)
        self.E = len(self.W[0])
                
    def dualGraph(self):
        W = self.W.T
        return hypergraph(W)

    def cliqueExpand(self):
        A = np.zeros((len(self.W), len(self.W)))
        for e in range(len(self.W[0])):
            vxc = np.where(self.W[:, e] == 1)[0]
            for i in range(len(vxc)):
                for j in range(i+1, len(vxc)):
                    A[vxc[i], vxc[j]] = 1
        A = A + A.T
        return G(A)

    def lineGraph(self):
        return self.cliqueGraph(self.dualGraph())

    def starGraph(self):
        A = np.zeros((len(self.W) + len(self.W[0]), len(self.W) + len(self.W[0])))
        for e in range(len(self.W[0])):
            vxc = np.where(self.W[:, e] == 1)[0]
            A[vxc, len(self.W) + e] = 1
        A = A + A.T
        return G(A)

    def bollaLaplacian(self):
        print('function stub')

    def rodriguezLaplacian(self):
        print('function stub')

    def zhouLaplacian(self):
        print('function stub')
        
    def adjacencyTensor(self):
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
        order = sum(self.W[:,0])
        modes = len(self.W) * np.ones(order)
        D = np.zeros(modes.astype(int))
        for vx in range(len(W)):
            D[(np.ones(order) * vx).astype(int)] = sum(W[vx])
        return D
    
    def laplacianTensor(self)
        D = self.degreeTensor()
        A = self.adjacencyTensor()
        L = D - A
        return L
    