import numpy as np
import scipy as sp

class graph:

    def __init__(self):
        self.A = np.zeros((0,0))
        self.N = 0
        self.E = 0
        
    def __init__(self, A):
        self.A = A
        self.N = len(A)

    def avgDistance(self):
        print('Function stub')
