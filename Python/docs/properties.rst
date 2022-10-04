Hypergraph Properties
=====================
HAT implements computations for a series of standard graph and hypergraph theoretic properties. These properties are computed on
either hypergraph or graph objects, which are created based on incidence and adjacency matrices respectively.

.. code-block:: Python

    H = HAT.hypergraph.hypergraph(W)    # Create a hypergraph with an incidence matrix W
    G = HAT.graph.graph(A)              # Create a graph with an adjacency matrix A

Diameter
********
The diameter is defined as the maximum minimum distance between any two vertices in a graph or hypergraph.

.. code-block:: Python

    diameter = H.diameter

Clustering Coefficient
**********************
The clustering coefficient of a graph or hypergraph is the average clustering coefficient of all vertices. For
any given vertex, the vertex clustering coefficient is calculated as 

.. code-block:: Python

    gamma = H.clusteringCoefficient     # Hypergraph clustering coeffficient
    gammaI = H.clusteringCoefficient(i) # Clustering coefficient of vertex i

Average Distance
****************
The average distance is the pairwise distance between any two vertices.

.. code-block:: Python

    avgDistane = H.averageDistance

1. Diameter
2. Clustering Coefficient
3. Average Distance
4. etc.
