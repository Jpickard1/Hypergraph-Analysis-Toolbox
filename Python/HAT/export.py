import pandas as pd

def to_hypernetx(HG):
    """
    Converts a HAT hypergraph to a HyperNetX hypergraph.

    :param HG: The hypergraph object from the HAT library.
    :type HG: HAT.Hypergraph
    :return: A HyperNetX hypergraph object.
    :rtype: hypernetx.Hypergraph

    This function converts a hypergraph represented in the HAT library 
    to the HyperNetX library's `Hypergraph` format. It creates the HyperNetX 
    hypergraph using the node-edge relationships defined in the HAT hypergraph.

    **Parameters**
        - **HG** (*HAT.Hypergraph*): The input hypergraph in HAT format.

    **Returns**
        - (*hypernetx.Hypergraph*): The hypergraph in HyperNetX format.

    **Features**
        - Explodes the `edges` DataFrame to create a long-form representation of 
          the hypergraph, where each row associates a node with an edge.
        - Automatically assigns a column for edge identifiers (`Edges`) to maintain 
          the edge-node mappings required by HyperNetX.

    **Dependencies**
        This function requires the `hypernetx` library. If the library is not 
        installed, an `ImportError` will be raised with instructions to install it.

    **Raises**
        - `ImportError`: If the `hypernetx` library is not installed.

    **Notes**
        - Ensure that the input HAT hypergraph has a well-structured `edges` DataFrame 
          with a `Nodes` column containing lists of nodes in each edge.
        - The output HyperNetX hypergraph will preserve the original node and edge 
          relationships from the input HAT hypergraph.
    """
    try:
        import hypernetx as hnx
    except ImportError as e:
        raise ImportError(
            "The 'hypernetx' library is required to use the 'to_hypernetx' function. "
            "Please install it using `pip install hypernetx`."
        ) from e
    
    HG_df = HG.edges.explode('Nodes')
    HG_df['Edges'] = HG_df.index
    return hnx.Hypergraph(HG_df, edge_col='Edges', node_col='Nodes')

def to_hypergraphx(HG):
    """
    Converts a HAT hypergraph to a HypergraphX hypergraph.

    :param HG: The hypergraph object from the HAT library.
    :type HG: HAT.Hypergraph
    :return: A HypergraphX hypergraph object.
    :rtype: hypergraphx.core.hypergraph.Hypergraph

    This function converts a hypergraph represented in the HAT library 
    to the HypergraphX library's `Hypergraph` format. It transfers node and edge 
    metadata, as well as weights if the original hypergraph is weighted.

    **Parameters**
        - **HG** (*HAT.Hypergraph*): The input hypergraph in HAT format.

    **Returns**
        - (*hypergraphx.core.hypergraph.Hypergraph*): The hypergraph in HypergraphX format.

    **Features**
        - If the HAT hypergraph contains a `Weights` column in its `edges` DataFrame, 
          these weights are transferred to the HypergraphX hypergraph.
        - Transfers node metadata from the `nodes` DataFrame as a dictionary where 
          keys are node identifiers and values are metadata dictionaries.
        - Transfers edge metadata from the `edges` DataFrame as a dictionary where 
          keys are edge identifiers and values are metadata dictionaries.

    **Dependencies**
        This function requires the `hypergraphx` library. If the library is not 
        installed, an `ImportError` will be raised with instructions to install it.

    **Raises**
        - `ImportError`: If the `hypergraphx` library is not installed.

    **Notes**
        - Ensure that the input HAT hypergraph has properly defined `nodes` 
          and `edges` DataFrames with necessary metadata if applicable.
    """
    try:
        from hypergraphx.core import hypergraph as hgx
    except ImportError as e:
        raise ImportError(
            "The 'hypergraphx' library is required to use the 'to_hypergraphx' function. "
            "Please install it using `pip install hypergraphx`."
        ) from e
    weighted = ('Weights' in HG.edges.columns)
    weights = None
    if weighted:
        weights = list(HG.edges['Weights'].values)
    G = hgx.Hypergraph(
        HG.edge_list,
        weighted = weighted,
        weights = weights,
        node_metadata=HG.nodes.to_dict(orient='index'),
        edge_metadata=HG.edges.to_dict(orient='index') 
    )
    return G

def to_hif(HG):
    """Converts HAT.Hypergraph to Hypergraph Interchange Format (HIF)
    """
    network_type = 'HAT'
    metadata = {
        'uniform' : HG.uniform,
        'order' : HG.order,
        'directed' : HG.directed
    }
    incidence, nodes, edges = [], [], []
    for inode in range(HG.nodes.shape[0]):
        node = {'node': inode}
        for property in HG.nodes.columns:
            if property in ['Nodes']:
                continue
            else:
                node[property] = HG.nodes[property].iloc[inode]
        nodes.append(node)
    for iedge in range(HG.edges.shape[0]):
        incident_nodes = list(HG.edges['Nodes'].iloc[iedge])
        edge = {
            'edge' : iedge,
            'nodes' : incident_nodes
        }
        edge_incidence = {
            'edge' : iedge,
            'node' : None
        }
        for property in HG.edges.columns:
            if property in ['Nodes']:
                continue
            else:
                edge[property] = HG.edges[property].iloc[iedge]
                edge_incidence[property] = HG.edges[property].iloc[iedge]
        edges.append(edge)
        for node in incident_nodes:
            edge_incidence['node'] = node
            incidence.append(edge_incidence.copy())
    hif = {
        'network-type': network_type,
        'metadata': metadata,
        'nodes': nodes,
        'edges': edges,
        'incidences': incidence
    }
    return hif
    
