import numpy as np
import scipy as sp
from Hypergraph import hypergraph

class uniformHypergraph(hypergraph):

    def __init__(self):
        """
        The uniform hypergraph class defines hypergraph objects as a subclass
        of the hypergraph class. It contains member variables:
        - Incidence Matrix
        - Edge Set
        - k
        - N
        - E
        """
        self.W = np.zeros(0,0)
        self.EdgeSet = []
        self.k = 0
        self.N = 0
        self.E = 0

    def getIncidenceMatrix(self):
        return self.W

    def getEdgeSet(self):
        return self.EdgeSet

    def getK(self):
        return self.k

    def getN(self):
        return self.N

    def getE(self):
        return self.E

    def setIncidenceMatrix(self, w):
        self.W = w

    def setEdgeSet(self, es):
        self.EdgeSet = es

    def setK(self, K):
        self.k = K

    def setN(self, n):
        self.N = n

    def setE(self, e):
        self.E = e
