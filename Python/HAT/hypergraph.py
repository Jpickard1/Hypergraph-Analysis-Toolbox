# Incidence matrix and related manipulations
import numpy as np

# Adjacency tensor construction
from itertools import permutations

# User output
import warnings

# Hypergraph metadata
import pandas as pd

# Nice printing
from rich import print

# Possibly remove this dependency
import copy

# HAT modules
from HAT import graph
from HAT import export
from HAT import laplacian


class Hypergraph:
    """Represents a hypergraph structure, enabling complex multi-way relationships between nodes.

    This class implements a hypergraph where edges (or hyperedges) can connect multiple vertices. 
    Internally, the hypergraph can be represented by an incidence matrix, an adjacency tensor, or an 
    edge list, depending on the available inputs and the uniformity of the hypergraph.

    Formally, a hypergraph :math:`H=(V,E)` is defined by a set of vertices :math:`V` and a set of 
    edges :math:`E`, where each edge :math:`e \\in E` may contain any subset of vertices in :math:`V`. 
    Unlike traditional graphs, hypergraph edges can connect more than two vertices, allowing for 
    multi-way interactions. Uniform hypergraphs, where all edges have the same number of vertices, 
    can be efficiently represented using tensors.

    Attributes
    ----------
    edge_list : list of lists or None
        Stores the list of edges if provided during initialization.
    adjacency_tensor : ndarray or None
        Stores the adjacency tensor if provided during initialization.
    incidence_matrix : ndarray or None
        Stores the incidence matrix if generated or provided during initialization.
    nodes : ndarray or None
        Array of nodes (vertices) in the hypergraph.
    edges : ndarray or None
        Array of edges (hyperedges) in the hypergraph.
    uniform : bool
        Indicates if the hypergraph is uniform.
    k : int
        The degree of uniformity in terms of edge size, if applicable.
    directed : bool, default=False
        Indicates if the hypergraph is directed.
    verbose : bool, default=False
        Additional warning messages display when true
    """
    def __init__(self,
            edge_list=None,
            adjacency_tensor=None,
            incidence_matrix=None,
            nodes=None,
            edges=None,
            uniform=None,
            order=None,
            directed=None,
            compress=True,# this argument tells the constructor to rearrange the numerical 
                          # representation based on the set .edges dataframe.
            verbose=False
        ):
        # Auth: Joshua Pickard
        #       jpic@umich.edu
        # Date: Nov 30, 2022
        # print("Hypergraph constructor!")

        # Validate arguments
        self._validate_constructor_arguments(
            edge_list        = edge_list,
            adjacency_tensor = adjacency_tensor,
            incidence_matrix = incidence_matrix,
            nodes            = nodes,
            edges            = edges,
            uniform          = uniform,
            order            = order,
            directed         = directed
        )

        # Convert all nodes to integers, and build a map between integers and names
        if edge_list is not None:
            new_edge_list, name_to_int = convert_nodes_to_integers(edge_list)
            original_edge_list = copy.deepcopy(edge_list)
            edge_list = new_edge_list

        # Assign object data
        self._edge_list        = edge_list
        self._adjacency_tensor = adjacency_tensor
        self._incidence_matrix = incidence_matrix
        self._nodes            = nodes
        self._edges            = edges
        self._uniform          = uniform
        self._order            = order
        self._directed         = directed
        self._verbose          = verbose

        self._reset = {
            'adjacency tensor': True,
            'incidence matrix': True,
            'edge list': True
        }
        if self._adjacency_tensor is not None:
            self._reset['adjacency tensor'] = False
        if self._incidence_matrix is not None:
            self._reset['incidence matrix'] = False
        if self._edge_list is not None:
            self._reset['edge list'] = False

        # Set uniform and order of the hypergraph
        self._detect_uniform_and_order(uniform, order)

        # Set directed property of the hypergraph
        self._detect_directed(directed)

        # Set the nodes dataframe
        if self._nodes is None:
            flat_set = None
            if self._adjacency_tensor is not None:
                num_nodes = self._adjacency_tensor.shape[0]
            elif self._edge_list is not None:
                if isinstance(edge_list[0][0], list):
                    flat_list = [item for sublist in edge_list for subsublist in sublist for item in subsublist]
                    flat_set = list(set(flat_list))
                    num_nodes = len(flat_set)
                else:
                    flat_set = list(set([j for edge in edge_list for j in edge]))
                    num_nodes = len(flat_set)
            elif self._incidence_matrix is not None:
                num_nodes = self.incidence_matrix.shape[0]
            self._nodes = pd.DataFrame({'Nodes': np.arange(num_nodes)})
            if name_to_int is not None:
                int_to_name = {v: k for k, v in name_to_int.items()}
                node_labels = []
                for k in self._nodes['Nodes'].values:
                    node_labels.append(int_to_name[k])
                self._nodes['Names'] = node_labels
        elif 'Nodes' not in self._nodes.columns:
            self._nodes['Nodes'] = np.arange(self._nodes.shape[0])
            warnings.warn('`Nodes` column not found in the provided in self._nodes. It has been added') if self._verbose else None

        # Set the edges dataframe
        if self._edges is None:
            # Determine the number of edges, the nodes, the head, and the tail
            num_edges, edge_nodes, head, tail = 0, [], [], []
            if self._edge_list is not None:
                num_edges = len(self._edge_list)
                if self._directed:
                    for edge in self._edge_list:
                        edge_nodes.append(sorted(edge[0] + edge[1]))
                        head.append(sorted(edge[0]))
                        tail.append(sorted(edge[1]))
                else:
                    edge_nodes = self._edge_list
            elif self._incidence_matrix is not None:
                num_edges = self.incidence_matrix.shape[1]
                for iedge in range(num_edges):
                    edge_nodes.append(np.where(self._incidence_matrix[:,iedge] != 0)[0])
                    if self._directed:
                        head.append(np.where(self._incidence_matrix[:,iedge] > 0)[0])
                        tail.append(np.where(self._incidence_matrix[:,iedge] < 0)[0])
            elif self._adjacency_tensor is not None:
                compress = True
                idxs = np.where(self._adjacency_tensor != 0) 
                head = [[node] for node in list(idxs[0])]
                for iedge in range(len(idxs[0])):
                    tail.append(sorted([idxs[k][iedge] for k in range(1,len(idxs))]))
                    edge_nodes.append(sorted([head[iedge][0]] + tail[-1]))
                num_edges = len(head)
            if self.directed:
                self._edges = pd.DataFrame(
                    {
                        'Nodes' : edge_nodes,
                        "head"  : head,
                        "tail"  : tail
                    }
                )
            else:
                self._edges = pd.DataFrame(
                    {
                        'Nodes' : edge_nodes,
                    }
                )
        if 'Edges' not in self._edges.columns:
            self.edges['Edges'] = list(np.arange(self._edges.shape[0]))
            # = pd.DataFrame({'Edges': list(np.arange(self._edges.shape[0]))})
            warnings.warn('"Edges" column not found in the provided nodes dataframe.') if self._verbose else None
            warnings.warn('This column has been appended.') if self._verbose else None

        if compress:
            # TODO: filter self._edges to achieve:
            #   1. no duplicate rows (done)
            #   2. no duplicate tails (if directed)
            # Identify duplicate rows based on string representations
            self._edges['row_hash'] = self._edges.drop(["Edges"], axis=1).apply(lambda row: str(row.values), axis=1)
            self._edges = self._edges.drop_duplicates(subset=['row_hash']).drop(columns=['row_hash'])
            self.edges['Edges'] = list(np.arange(self._edges.shape[0]))
            # reset the incidence matrix, edge list, or adjacency tensor based on self.edges
            if incidence_matrix is not None:
                self._set_incidence_matrix()
            if edge_list is not None:
                self._set_edge_list()
            if adjacency_tensor is not None:
                self._set_adjacency_tensor()

    def _validate_constructor_arguments(
        self,
        edge_list=None,
        adjacency_tensor=None,
        incidence_matrix=None,
        nodes=None,
        edges=None,
        uniform=None,
        order=None,
        directed=None
    ):
        '''
        This appears to do nothing and can be killed
        '''
        # At least one numerical representation should be supplied
        if edge_list is None and adjacency_tensor is None and incidence_matrix is None and edges is None:
            warnings.warn("The edge list, incidence matrix, adjacency tensor, and edge DataFrame are None.", UserWarning) if self._verbose else None

        if order is not None and not (uniform == True):
            warnings.warn("If the hypergraph order is fixed then it should be k-uniform") if self._verbose else None

    def _detect_uniform_and_order(self, uniform, order):
        if self._adjacency_tensor is not None:
            self._uniform = True
            self._order = len(self._adjacency_tensor.shape)
        elif self._incidence_matrix is not None:
            nonzero_counts = np.count_nonzero(self._incidence_matrix, axis=0)
            self._uniform = np.all(nonzero_counts == nonzero_counts[0])
            self._order = nonzero_counts[0]
        elif self._edge_list is not None:
            # Determine if the hypergraph is directed
            if not isinstance(self._edge_list[0][0], list): # (undirected)
                k = len(self._edge_list[0])
                self._uniform = True
                self._order = k
                for e in self._edge_list:
                    if len(e) != k:
                        self._uniform = False
                        break
            else:  # (directed)
                k = len(self._edge_list[0][0])
                self._uniform = True
                self._order = k
                for e in self._edge_list:
                    if len(e[0]) != k:
                        self._uniform = False
                        break
        if self._uniform == False:
            self._order = -1

        if self._uniform != uniform and uniform is not None:
            warnings.warn('The provided and detected `uniform` are not in agreement!') if self._verbose else None

        if self._order != order and order is not None:
            warnings.warn('The provided and detected `order` are not in agreement!') if self._verbose else None

    def _detect_directed(self, directed):
        if self._edges is not None and "head" in self._edges.columns and "tail" in self._edges.columns:
            self._directed = True

        # The adjacency tensor is directed
        elif self._adjacency_tensor is not None:
            self._directed = True

        # The indicence matrix is directed if there are +/- terms
        elif self._incidence_matrix is not None and np.sum(self._incidence_matrix > 0) > 0 and np.sum(self._incidence_matrix < 0) > 0:
            self._directed = True

        # The edge list is directed if an edge has the form: [[head], [tail]]
        elif self._edge_list is not None:
            self._directed = True
            for edge in self._edge_list:
                if not (len(edge) >= 2 and isinstance(edge[0], list) and isinstance(edge[1], list)):
                    self._directed = False

        # By default, the system is undirected
        else:
            self._directed = False

        if self._directed != directed and directed is not None:
            warnings.warn('The provided and detected `directed` are not in agreement!') if self._verbose else None

    @property
    def nodes(self):
        return self._nodes
        
    @property
    def edges(self):
        return self._edges

    @property
    def directed(self):
        return self._directed

    @property
    def nnodes(self):
        """Returns the number of nodes (vertices) in the hypergraph.

        Returns
        -------
        int
            Number of nodes in the hypergraph.
        """
        return self._nodes.shape[0] if self._nodes is not None else 0

    @property
    def nedges(self):
        """Returns the number of edges (hyperedges) in the hypergraph.

        Returns
        -------
        int or bool
            Number of edges if available, otherwise False if edges are not defined.
        """
        return self.edges.shape[0]

    @property
    def edge_weights(self):
        if 'weight' not in self.edges.columns:
            self.edges['weight'] = np.ones((self.nedges,))
        return np.array(self.edges['weight'].values)

    @property
    def node_weights(self):
        if 'weight' not in self.nodes.columns:
            self.nodes['weight'] = np.ones((self.nnodes,))
        return np.array(self.nodes['weight'].values)

    @property
    def uniform(self):
        """Returns if the hypergraph was uniform."""
        return self._uniform
    
    @property
    def order(self):
        return self._order
    
    @property
    def incidence_matrix(self):
        """Returns the incidence matrix of the hypergraph, constructing it if necessary."""
        if self._incidence_matrix is None or self._reset['incidence matrix']:
            self._set_incidence_matrix()  # Automatically set if not yet defined
        self._reset['incidence matrix'] = False
        return self._incidence_matrix

    @property
    def edge_list(self):
        """Returns the edge list of the hypergraph, constructing it if necessary."""
        if self._edge_list is None or self._reset['edge list']:
            self._set_edge_list()  # Automatically set if not yet defined
        self._reset['edge list'] = False
        return self._edge_list

    @property
    def adjacency_tensor(self):
        """Returns the adjacency tensor of the hypergraph, constructing it if necessary."""
        if self._adjacency_tensor is None or self._reset['adjacency tensor']:
            self._set_adjacency_tensor()  # Automatically set if not yet defined
        self._reset['adjacency tensor'] = False
        return self._adjacency_tensor
    
    @property
    def star_graph(self):
        return graph.star_graph(self)
    
    @property
    def clique_graph(self):
        return graph.clique_graph(self)
    
    @property
    def laplacian_matrix(self, laplacian_type='bolla'):
        return laplacian.laplacian_matrix(self, laplacian_type=laplacian_type)

    def _set_incidence_matrix(self):
        """Sets self._incidence_matrix based on self._edges
        """
        incidence_matrix = np.zeros((self.nnodes, self.nedges))
        for iedge in range(self.nedges):
            if self.directed:
                tail = self.edges["tail"].iloc[iedge]
                head = self.edges["head"].iloc[iedge]
                for node in tail:
                    incidence_matrix[node, iedge] = -1
                for node in head:
                    incidence_matrix[node, iedge] = 1
            else:
                nodes = self.edges['Nodes'].iloc[iedge]
                for node in nodes:
                    incidence_matrix[node, iedge] = 1
        self._incidence_matrix = incidence_matrix

    def _set_edge_list(self):
        """Sets self._edge_lsit based on self._edges
        """
        edge_list = []
        if self.directed:
            for iedge in range(self.nedges):
                edge_list.append(list([self.edges["head"].iloc[iedge], self.edges["tail"].iloc[iedge]]))
        else:
            for iedge in range(self.nedges):
                edge_list.append(list(self.edges['Nodes'].iloc[iedge]))
        self._edge_list = edge_list

    def _set_adjacency_tensor(self):
        """Sets self._adjacency_tensor based on self._edges
        """
        adjacency_tensor = np.zeros((self.nnodes,) * self.order, dtype=int)
        if self.directed:
            for iedge in range(self.nedges):
                tail = self.edges["tail"].iloc[iedge]
                head = self.edges["head"].iloc[iedge]
                if not isinstance(head, list):
                    head = [head]
                if not isinstance(tail, list):
                    tail = [tail]
                for head_node in head:
                    for perm in permutations(tail):
                        adjacency_tensor[tuple([head_node] + list(perm))] = 1
        else:
            for iedge in range(self.nedges):
                edge_nodes = self.edges['Nodes'].iloc[iedge]
                for perm in permutations(edge_nodes):
                    adjacency_tensor[tuple(perm)] = 1

        self._adjacency_tensor = adjacency_tensor

    def _remove_duplicates(self, list_of_lists):
        if isinstance(list_of_lists[0][0], int):
            return list_of_lists
        # Use a set to track unique items
        unique_items = set()
        deduplicated_list = []

        for item in list_of_lists:
            # Convert the two inner lists to a tuple of sorted tuples
            normalized_item = tuple(sorted(map(tuple, item)))
            if normalized_item not in unique_items:
                unique_items.add(normalized_item)
                deduplicated_list.append(item)

        return deduplicated_list

    def add_node(self, properties=None):
        """
        Adds a new node to `self._nodes`. Properties is a dictionary specifying
        values for the columns in `self._nodes`. Missing properties are filled
        with `None`.

        Parameters:
            properties (dict, optional): A dictionary containing column names as keys
                                        and their corresponding values. Defaults to None.
        """
        if properties is None:
            properties = {}
        properties['Nodes'] = self.nnodes

        # Convert properties to a DataFrame with one row
        new_row = pd.DataFrame([properties])

        # Align new_row columns with self._nodes, filling missing columns with None
        self._nodes = pd.concat([self._nodes, new_row])
        self._nodes.reset_index(drop=True, inplace=True)

        # Representations must be reset based on the dataframes
        for key in self._reset.keys():
            self._reset[key] = True

    def add_edge(self, nodes, properties=None):

        # Validate nodes
        if isinstance(nodes[0], list):
            for node in nodes[0] + nodes[1]:
                if node not in self.nodes.index:
                    warnings.warn(f'Node {node} not found in the hypergraph') if self._verbose else None
        else:
            for node in nodes:
                if node not in self.nodes.index:
                    warnings.warn(f'Node {node} not found in the hypergraph') if self._verbose else None

        # Initialize properties if None
        properties = properties or {}

        # Handle directed vs. undirected edges
        if self.directed:
            if len(nodes) < 2:
                raise ValueError("Directed edges require at least two nodes (Head and Tail).")
            properties["head"] = nodes[0]
            properties["tail"] = nodes[1]
            properties['Nodes'] = nodes[0] + nodes[1]
        else:
            properties['Nodes'] = nodes

        properties['Edges'] = self.nedges

        # Append the edge to the DataFrame
        new_row = pd.DataFrame([properties])
        self._edges = pd.concat([self._edges, new_row], ignore_index=True)
        self._edges.reset_index(drop=True, inplace=True)

        # Representations must be reset based on the dataframes
        for key in self._reset.keys():
            self._reset[key] = True

    @property
    def dual(self):
        """The dual hypergraph is constructed.
        Let :math:`H=(V,E)` be a hypergraph. In the dual hypergraph each original edge :math:`e in E`
        is represented as a vertex and each original vertex :math:`v in E` is represented as an edge. Numerically, the
        transpose of the incidence matrix of a hypergraph is the incidence matrix of the dual hypergraph.
        
        :return: Hypergraph object
        :rtype: *Hypergraph*

        References
        ----------
          - Yang, Chaoqi, et al. "Hypergraph learning with line expansion." arXiv preprint arXiv:2005.04843 (2020).
        """
        # Auth: Joshua Pickard
        #       jpic@umich.edu
        # Date: Nov 30, 2022
        return Hypergraph(incidence_matrix=self.incidence_matrix.T)

    @property
    def hif(self):
        export.to_hif(self)

    @property
    def hypernetx(self):
        return export.to_hypernetx(self)

    @classmethod
    def from_hypernetx(cls, HG_hnx):
        node_df = pd.DataFrame(
            {
                'Nodes' : np.arange(HG_hnx.dataframe['nodes'].nunique()),
                'Names': list(HG_hnx.dataframe['nodes'].unique())
            }
        )
        edge_list, node_names, weight, properties = [], [], [], []
        for _, df in HG_hnx.dataframe.groupby('edges'):
            node_names.append(list(df['nodes'].unique()))
            idxs = []
            for node in node_names[-1]:
                idxs.append(np.where(node_df['Names'] == node)[0][0])
            edge_list.append(idxs)
            if 'weight' in df.columns:
                weight.append(df['weight'].iloc[0])
            else:
                weight.append(1)
        edge_df = pd.DataFrame(
            {
                'Edges' : np.arange(len(edge_list)),
                'Nodes' : edge_list,
                'Node Names' : node_names,
                'Weight': weight
            }
        )

        HG = Hypergraph(
            edge_list = edge_list,
            nodes     = node_df,
            edges     = edge_df
        )
        return HG

    @classmethod
    def from_networkx(cls, nxg):
        import networkx as nx
        node_data = list(nxg.nodes(data=True)) 
        nodes_df = pd.DataFrame(node_data, columns=['node', 'attributes'])
        nodes_df['Nodes'] = [i for i in range(len(node_data))]
        nodes, head, tail = [], [], []
        edges_df = nx.to_pandas_edgelist(nxg)
        edge_list = []
        if isinstance(nxg, nx.classes.digraph.DiGraph):
            directed = True
            for edge in nxg.edges:
                i0 = nodes_df[nodes_df['node'] == edge[0]]['Nodes'].iloc[0]
                i1 = nodes_df[nodes_df['node'] == edge[1]]['Nodes'].iloc[0]
                tail.append(i0)
                head.append(i1)
                nodes.append([i0, i1])
                edge_list.append([[i0], [i1]])
            edges_df['head'] = head
            edges_df['tail'] = tail
        else:
            directed=False
            for edge in nxg.edges:
                i0 = nodes_df[nodes_df['node'] == edge[0]]['Nodes'].iloc[0]
                i1 = nodes_df[nodes_df['node'] == edge[1]]['Nodes'].iloc[0]
                nodes.append([i0, i1])
                edge_list.append([i0, i1])
        edges_df['Nodes'] = nodes
        HG = Hypergraph(
            nodes = nodes_df,
            edges = edges_df,
            edge_list=edge_list,
            compress=False,
            directed=directed
        )
        return HG

    @classmethod
    def from_hif(cls, hif):
        # Create nodes dataframe
        nodes = pd.DataFrame(hif['nodes'])
        if 'node' in nodes.columns:
            rename_column_nodes = list(nodes.columns).index('node')
            columns_nodes = list(nodes.columns)
            columns_nodes[rename_column_nodes] = 'Nodes'
            nodes.columns = columns_nodes

        # Create edges dataframe
        edges = pd.DataFrame(hif['edges'])
        if 'nodes' in edges.columns:
            rename_column_edges = list(edges.columns).index('nodes')
            columns_edges = list(edges.columns)
            columns_edges[rename_column_edges] = 'Nodes'
            edges.columns = columns_edges

        if 'attrs' in edges.columns:
            attrs_expanded = pd.json_normalize(edges['attrs'])
            edges = pd.concat([edges.drop(columns=['attrs']), attrs_expanded], axis=1)
        if 'attrs' in nodes.columns:
            attrs_expanded = pd.json_normalize(nodes['attrs'])
            nodes = pd.concat([nodes.drop(columns=['attrs']), attrs_expanded], axis=1)

        # Create the edge list
        edge_list = [[] for _ in range(edges.shape[0])]
        for incidence in hif['incidences']:
            edge = incidence['edge']
            node = incidence['node']
            edge_idx = list(edges['edge'].values).index(edge)
            node_idx = list(nodes['Nodes'].values).index(node)
            edge_list[edge_idx].append(node_idx)
        edges['Nodes'] = edge_list

        HG = Hypergraph(
            nodes = nodes,
            edges = edges,
            edge_list=edge_list,
            compress=False
        )
        return HG
    
    def to_hif(self):
        """Converts HAT.Hypergraph to Hypergraph Interchange Format (HIF)
        """
        network_type = 'undirected'
        metadata = {
            'uniform' : self.uniform,
            'order' : self.order,
            'directed' : self.directed
        }
        incidence, nodes, edges = [], [], []
        for inode in range(self.nodes.shape[0]):
            node = {
                'node': inode,
#                'weight':None,
                'attrs': {}
            }
            for property in self.nodes.columns:
                if property in ['Nodes']:
                    continue
                elif property == 'weight':
                    node['weight'] = self.nodes[property].iloc[inode]
                else:
                    node['attrs'][property] = self.nodes[property].iloc[inode]
            nodes.append(node)
        for iedge in range(self.edges.shape[0]):
            incident_nodes = list(self.edges['Nodes'].iloc[iedge])
            edge = {
                'edge' : iedge,
#                'weight': None,
                'attrs': {
                    'nodes' : incident_nodes
                }
            }
            edge_incidence = {
                'edge' : iedge,
                'node' : None,
#                'weight': None,
#                'direction': None,
                'attrs': {}
            }
            for property in self.edges.columns:
                if property in ['Nodes', "head", "tail"]:
                    continue
                if property == 'weight':
                    edge['weight'] = self.edges[property].iloc[iedge]
                    edge_incidence['weight'] = self.edges[property].iloc[iedge]
                else:
                    edge['attrs'][property] = self.edges[property].iloc[iedge]
                    edge_incidence['attrs'][property] = self.edges[property].iloc[iedge]
            if self.directed:
                edge['attrs']['head'] = self.edges['head'].iloc[iedge]
                edge['attrs']['tail'] = self.edges['tail'].iloc[iedge]
            edges.append(edge)
            for node in incident_nodes:
                edge_incidence['node'] = node
                if self.directed:
                    if node in edge['attrs']['head']:
                        edge_incidence['direction'] = 'head'
                    else:
                        edge_incidence['direction'] = 'tail'
                incidence.append(edge_incidence.copy())

        hif = {
            'network-type': network_type,
            'metadata': metadata,
            'nodes': nodes,
            'edges': edges,
            'incidences': incidence
        }

        return hif

    @property
    def hypergraphx(self):
        return export.to_hypergraphx(self)
    
def convert_nodes_to_integers(edge_list):
    # Step 1: Check if the lowest level contains integers
    # Flatten the edge_list and check if any of the elements are integers
    all_elements = [elem for sublist in edge_list for elem in sublist]
    
    # If all elements are integers, no need to do anything
    if all(isinstance(elem, int) for elem in all_elements):
        return edge_list, None  # No changes, return original edge_list

    # Step 2: Map each unique name to an integer if names are present
    # Create a mapping from names to unique integers
    unique_names = set(all_elements)
    name_to_int = {name: idx for idx, name in enumerate(unique_names)}

    # Step 3: Create a new edge_list with integer mappings
    new_edge_list = [[name_to_int[name] for name in edge] for edge in edge_list]

    # Step 4: Return the new edge_list and the mapping
    return new_edge_list, name_to_int
