import numpy as np
import matplotlib.pyplot as plt

from HAT.graph import graph
from HAT.hypergraph import hypergraph as HG

def IM(H):
    """
    Plot the incidence matrix of a hypergraph.
    """
    W = H.W
    y,x = np.where(W == 1)
    for i in range(len(W[0])):
        i_pts = np.where(x == i)
        plt.plot([i,i], [np.min(y[i_pts]), np.max(y[i_pts])], zorder=1)
    plt.tick_params(left = False, right = False, labelleft = False, labelbottom = False, bottom = False)
    plt.scatter(x,y, zorder=2)
    plt.title('Incidence Matrix')
