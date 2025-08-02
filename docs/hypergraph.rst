Hypergraph Class
~~~~~~~~~~~~~~~~~~~~~~

The ``Hypergraph`` class serves as the core data structure in HAT. It represents a hypergraph's nodes, hyperedges, and related metadata. It also supports a variety of numerical representations, including tensors and incidence matrices.

Creating a ``Hypergraph`` object provides access to additional analysis and visualization functionality across the toolbox’s modules.

.. currentmodule:: HAT

.. autoclass:: Hypergraph
   :show-inheritance:
   :noindex:

   .. rubric:: Constructor

   .. automethod:: __init__

   .. rubric:: Core Properties

   .. autosummary::
      :nosignatures:

      ~Hypergraph.nodes
      ~Hypergraph.edges
      ~Hypergraph.directed
      ~Hypergraph.uniform
      ~Hypergraph.order

   .. rubric:: Size Properties

   .. autosummary::
      :nosignatures:

      ~Hypergraph.nnodes
      ~Hypergraph.nedges

   .. rubric:: Weight Properties

   .. autosummary::
      :nosignatures:

      ~Hypergraph.node_weights
      ~Hypergraph.edge_weights

   .. rubric:: Representations

   .. autosummary::
      :nosignatures:

      ~Hypergraph.incidence_matrix
      ~Hypergraph.edge_list
      ~Hypergraph.adjacency_tensor

   .. rubric:: Derived Graphs

   .. autosummary::
      :nosignatures:

      ~Hypergraph.star_graph
      ~Hypergraph.clique_graph
      ~Hypergraph.connected_components
      ~Hypergraph.dual

   .. rubric:: Mathematical Properties

   .. autosummary::
      :nosignatures:

      ~Hypergraph.laplacian_matrix

   .. rubric:: Export Properties

   .. autosummary::
      :nosignatures:

      ~Hypergraph.hif
      ~Hypergraph.hypergraphx

   .. rubric:: Methods

   .. autosummary::
      :nosignatures:

      ~Hypergraph.add_node
      ~Hypergraph.add_edge
      ~Hypergraph.to_hif

   .. rubric:: Class Methods

   .. autosummary::
      :nosignatures:

      ~Hypergraph.from_networkx
      ~Hypergraph.from_hif