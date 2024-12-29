import numpy as np
import networkx as nx

from scipy.spatial.distance import pdist

import matplotlib.pyplot as plt
from matplotlib.collections import PolyCollection
from matplotlib import colormaps
from itertools import combinations
from collections import Counter


from HAT.export import to_hypernetx
# from HAT.graph import graph
# from HAT.Hypergraph import Hypergraph as HG

def bipartite(
    HG,
    ax=None
):
    G = HG.star_graph
    pos = nx.layout.bipartite_layout(G, nodes=np.arange(HG.nedges))
    ax = ax or plt.gca()
    nx.draw(G, pos=pos, ax=ax)
    return ax

def pairwise(
    HG,
    ax=None
):
    G = HG.clique_graph
    pos = nx.layout.spring_layout(G)
    ax = ax or plt.gca()
    nx.draw(G, pos=pos)
    return ax


def clique(HG, ax=None, node_size=20, marker='o', cmap='viridis', edgewidth=5):
    """
    Plot the clique graph of a hypergraph, with edges of each clique sharing the same color.
    
    Parameters:
    - HG: Hypergraph object with attributes `clique_graph` (Graph) and `edges['Nodes']` (DataFrame).
    - ax: matplotlib.axes.Axes, optional. Axes to plot on.
    - node_size: int, optional. Size of the scatter plot points (default: 10).
    - marker: str, optional. Marker style for scatter plot points (default: 'o').
    - cmap: str or Colormap, optional. Colormap for clique edges (default: 'viridis').

    Returns:
    - ax: matplotlib.axes.Axes with the plot.
    """
    G = HG.clique_graph
    pos = np.array(list(nx.layout.spring_layout(G).values()))  # Positions as an array

    ax = ax or plt.gca()
    cmap = colormaps[cmap] if isinstance(cmap, str) else cmap  # Resolve colormap

    edge_counter = Counter()
    for nodes in HG.edges['Nodes']:
        for nodei, nodej in combinations(nodes, 2):
            edge_counter[tuple(sorted((nodei, nodej)))] += edgewidth

    for edge_idx, nodes in enumerate(HG.edges['Nodes']):
        color = cmap(edge_idx / HG.nedges)  # Normalize edge_idx to [0, 1] for colormap
        for nodei, nodej in combinations(nodes, 2):  # All pairs in the clique
            linewidth = edge_counter[tuple(sorted((nodei, nodej)))]
            edge_counter[tuple(sorted((nodei, nodej)))] -= edgewidth
            ax.plot(
                [pos[nodei, 0], pos[nodej, 0]],
                [pos[nodei, 1], pos[nodej, 1]],
                color=color,
                linewidth=linewidth,
                zorder=edge_idx,
            )

    # Scatter plot for nodes
    ax.scatter(
        pos[:, 0],
        pos[:, 1],
        s=node_size,
        marker=marker,
        zorder=HG.nedges + 1,
        color="black"
    )

    # Remove axis ticks
    ax.set_xticks([])
    ax.set_yticks([])
    return ax



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

def rubber_bands(
    HG,
    pos=None,
    with_color=True,
    with_node_counts=False,
    with_edge_counts=False,
    layout=nx.spring_layout,
    layout_kwargs={},
    ax=None,
    node_radius=None,
    edges_kwargs={},
    nodes_kwargs={},
    edge_labels_on_edge=True,
    edge_labels={},
    edge_labels_kwargs={},
    node_labels={},
    node_labels_kwargs={},
    with_edge_labels=True,
    with_node_labels=True,
    node_label_alpha=0.35,
    edge_label_alpha=0.35,
    with_additional_edges=None,
    contain_hyper_edges=False,
    additional_edges_kwargs={},
    return_pos=False
):
    try:
        import hypernetx as hnx
    except ImportError as e:
        raise ImportError(
            "The 'hypernetx' library is required to use the 'rubber_bands' function. "
            "Please install it using `pip install hypernetx`."
        ) from e
    HG_hnx = to_hypernetx(HG)
    hnx.draw(
        HG_hnx,
        pos=pos,
        with_color=with_color,
        with_node_counts=with_node_counts,
        with_edge_counts=with_edge_counts,
        layout=layout,
        layout_kwargs=layout_kwargs,
        ax=ax,
        node_radius=node_radius,
        edges_kwargs=edges_kwargs,
        nodes_kwargs=nodes_kwargs,
        edge_labels_on_edge=edge_labels_on_edge,
        edge_labels=edge_labels,
        edge_labels_kwargs=edge_labels_kwargs,
        node_labels=node_labels,
        node_labels_kwargs=node_labels_kwargs,
        with_edge_labels=with_edge_labels,
        with_node_labels=with_node_labels,
        node_label_alpha=node_label_alpha,
        edge_label_alpha=edge_label_alpha,
        with_additional_edges=with_additional_edges,
        contain_hyper_edges=contain_hyper_edges,
        additional_edges_kwargs=additional_edges_kwargs,
        return_pos=return_pos
    )

