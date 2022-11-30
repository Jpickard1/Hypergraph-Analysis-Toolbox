import numpy as np
import matplotlib.pyplot as plt

# from HAT.graph import graph
# from HAT.Hypergraph import Hypergraph as HG

def incidencePlot(H, shadeRows=True, connectNodes=True, dpi=200, edgeColors=None):
    """Plot the incidence matrix of a hypergraph.
    
    :param H: a HAT.hypergraph object
    :param shadeRows: shade rows (bool)
    :param connectNodes: connect nodes in each hyperedge (bool)
    :param dpi: the resolution of the image (int)
    :param edgeColors: The colors of edges represented in the incidence matrix. This is random by default
    
    :return: matplotlib axes with figure drawn on to it
    """
    # dpi spec
    plt.rcParams['figure.dpi'] = dpi
    
    n, m = H.IM.shape
    
    # plot the incidence matrix
    y, x = np.where(H.IM == 1)
    plt.scatter(x, y, 
                edgecolor='k',
                zorder=2)
    
    for i in range(m):
        y = np.where(H.IM[:,i] == 1)[0]
        x = i * np.ones(len(y),)
        if edgeColors is None:
            c = None
        else:
            c = edgeColors[i]
        plt.scatter(x, y, 
                    color=c,
                    edgecolor='k',
                    zorder=2)     

    y, x = np.where(H.IM == 1)
    
    # create row shading
    if shadeRows:
        yBar = np.arange(n)
        xBar = np.zeros(n)
        xBar[::2] = m - 0.5

        plt.barh(yBar, 
                 xBar, 
                 height=1.0,
                 color='grey', 
                 left=-0.25,
                 alpha=0.5,
                 zorder=1)
    
    # plot each hyperedge with a black connector
    if connectNodes:
        for i in range(len(H.IM[0])):
            i_pts = np.where(x == i)
            plt.plot([i,i], 
                     [np.min(y[i_pts]), 
                      np.max(y[i_pts])], 
                      c='k',
                      lw=1,
                      zorder=1)
    
    # plot range spec
    plt.xlim([-0.5, m - 0.5])   
    
    # Turn of axis ticks. Keep labels on
    plt.yticks([])
    plt.xticks([])
    
    return plt.gca()
