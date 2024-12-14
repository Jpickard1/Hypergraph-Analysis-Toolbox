import numpy as np
import matplotlib.pyplot as plt

# from HAT.graph import graph
# from HAT.Hypergraph import Hypergraph as HG

def incidence_plot(HG, shade_rows=True, connect_nodes=True, dpi=200, edge_colors=None):
    """Plot the incidence matrix of a hypergraph.
    
    :param H: a HAT.hypergraph object
    :param shade_rows: shade rows (bool)
    :param connect_nodes: connect nodes in each hyperedge (bool)
    :param dpi: the resolution of the image (int)
    :param edge_colors: The colors of edges represented in the incidence matrix. This is random by default
    
    :return: matplotlib axes with figure drawn on to it
    """
    # dpi spec
    plt.rcParams['figure.dpi'] = dpi
    
    n, m = HG.nnodes, HG.nedges
    
    # plot the incidence matrix
    y, x = np.where(HG.incidence_matrix != 0)
    plt.scatter(x, y, 
                edgecolor='k',
                zorder=2)
    
    for i in range(m):
        y = np.where(HG.incidence_matrix[:,i] != 0)[0]
        x = i * np.ones(len(y),)
        if edge_colors is None:
            c = None
        else:
            c = edge_colors[i]
        plt.scatter(x, y, 
                    color=c,
                    edgecolor='k',
                    zorder=2)     

    y, x = np.where(HG.incidence_matrix != 0)
    
    # create row shading
    if shade_rows:
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
    if connect_nodes:
        for i in range(len(HG.incidence_matrix[0])):
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
