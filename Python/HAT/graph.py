import numpy as np
import scipy as sp

class graph:

    def __init__(self):
        print('HAT/graph.py')
        self.A = np.zeros((0,0))
        self.N = 0
        self.E = 0

    def getIncidenceMatrix(self):
        return self.W

    def getN(self):
        return self.N

    def getE(self):
        return self.E

    def setIncidenceMatrix(self, w):
        self.W = w

    def setN(self, n):
        self.N = n

    def setE(self, e):
        self.E = e

    def plot(self):
        print('function stub')

    def cliqueExpand(self):
        print(self)
        print('function called')
