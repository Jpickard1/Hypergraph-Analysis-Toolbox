import numpy as np
import scipy as sp
from HAT.graph import graph as G

class hypergraph:

    def __init__(self):
        print('HAT/hypergraph.py')
        self.W = np.zeros((0,0))
        self.N = 0
        self.E = 0

    def getIncidenceMatrix(self):
        return self.W

    def getN(self):
        return self.N

    def getE(self):
        return self.E

    def setIncidenceMatrix(self, w):
        self.setN(len(w))
        self.setE(len(w[0]))
        self.W = w

    def setN(self, n):
        self.N = n

    def setE(self, e):
        self.E = e

    def plot(self):
        print('function stub')

    def cliqueExpand(self):
        g = G()
        print(self)
        print(g)
        print('function called')

    def lineGraph(self):
        print('function stub')

    def starGraph(self):
        print('function stub')

    def bollaLaplacian(self):
        print('function stub')

    def rodriguezLaplacian(self):
        print('function stub')

    def zhouLaplacian(self):
        print('function stub')