import sys
# sys.path.append('../')
from HAT import Hypergraph
import numpy as np
import pandas as pd
from pandas.testing import assert_frame_equal
import unittest
import logging

def are_nested_lists_equivalent(list1, list2):
    # Check if both are lists of lists
    if not all(isinstance(sublist, list) for sublist in list1 + list2):
        return False

    # Sort each sublist and then sort the outer list
    sorted_list1 = sorted([sorted(sublist) for sublist in list1])
    sorted_list2 = sorted([sorted(sublist) for sublist in list2])

    # Compare the sorted lists
    return sorted_list1 == sorted_list2

class HypergraphConstructorTestCase1(unittest.TestCase):
    """Tests hypergraph construction from three basic, numerical representations of the hypergraph.
    """
    def setUp(self):
        """Set up shared resources for edge_list-based tests."""
        self.edge_list = [[0, 1, 2], [0, 1, 3]]
        self.expected_node_df = pd.DataFrame({'Nodes': [0, 1, 2, 3]})
        self.expected_edges_df = pd.DataFrame(
            {
                'Nodes': [[0,1,2],
                          [0,1,3]]
            }
        )
        self.expected_incidence_matrix = np.array([
            [1, 1],
            [1, 1],
            [1, 0],
            [0, 1]
        ])
        self.expected_adjacency_tensor = np.array([
            [[0, 0, 0, 0],
             [0, 0, 1, 1],
             [0, 1, 0, 0],
             [0, 1, 0, 0]],
            [[0, 0, 1, 1],
             [0, 0, 0, 0],
             [1, 0, 0, 0],
             [1, 0, 0, 0]],
            [[0, 1, 0, 0],
             [1, 0, 0, 0],
             [0, 0, 0, 0],
             [0, 0, 0, 0]],
            [[0, 1, 0, 0],
             [1, 0, 0, 0],
             [0, 0, 0, 0],
             [0, 0, 0, 0]]
        ])

    def test_constructor_edge_set_1(self):
        """
        Test construction from edge_list.

        Validation of 3 numerical representations and other properties
        """
        logging.info('test_constructor_edge_set_1')
        HG = Hypergraph(edge_list=self.edge_list)
        incidence_matrix=np.array(
            [[1, 1],
            [1, 1],
            [1, 0],
            [0, 1]]
        )

        # Validate hypergraph properties
        assert HG.order == 3
        assert HG.uniform == True
        assert HG.directed == False
        np.testing.assert_array_equal(HG.incidence_matrix, incidence_matrix)
        np.testing.assert_array_equal(HG.adjacency_tensor, self.expected_adjacency_tensor)

        # Validate dataframes
        try:
            assert_frame_equal(self.expected_node_df, HG.nodes, check_dtype=False)
        except AssertionError as e:
            logging.info(f"DataFrames are not equal: {e}")

        try:
            assert_frame_equal(self.expected_edges_df, HG.edges, check_dtype=False)
        except AssertionError as e:
            logging.info(f"DataFrames are not equal: {e}")

        assert are_nested_lists_equivalent(HG.edge_list, self.edge_list) == True
        assert np.sum(HG.adjacency_tensor - self.expected_adjacency_tensor) < 1e-5

        logging.info('constructor_1 complete')

    def test_constructor_incidence_matrix_1(self):
        """
        Test construction from incidence_matrix.

        Validation of 3 numerical representations and other properties
        """
        edge_list = [[0,1,2],
                    [0,1,3]]
        incidence_matrix=np.array(
            [[1, 1],
            [1, 1],
            [1, 0],
            [0, 1]]
        )
        adjacency_tensor = np.array(
            [[[0, 0, 0, 0],
            [0, 0, 1, 1],
            [0, 1, 0, 0],
            [0, 1, 0, 0]],
            [[0, 0, 1, 1],
            [0, 0, 0, 0],
            [1, 0, 0, 0],
            [1, 0, 0, 0]],
            [[0, 1, 0, 0],
            [1, 0, 0, 0],
            [0, 0, 0, 0],
            [0, 0, 0, 0]],
            [[0, 1, 0, 0],
            [1, 0, 0, 0],
            [0, 0, 0, 0],
            [0, 0, 0, 0]]]
        )
        HG = Hypergraph(incidence_matrix=incidence_matrix)

        # Validate hypergraph properties
        assert HG.order == 3
        assert HG.uniform == True
        assert HG.directed == False
        np.testing.assert_array_equal(HG.incidence_matrix, incidence_matrix)
        np.testing.assert_array_equal(HG.adjacency_tensor, adjacency_tensor)

        # Validate dataframes
        try:
            assert_frame_equal(self.expected_node_df, HG.nodes, check_dtype=False)
        except AssertionError as e:
            logging.info(f"DataFrames are not equal: {e}")

        try:
            assert_frame_equal(self.expected_edges_df, HG.edges, check_dtype=False)
        except AssertionError as e:
            logging.info(f"DataFrames are not equal: {e}")

        assert are_nested_lists_equivalent(HG.edge_list, edge_list) == True
        assert np.sum(HG.adjacency_tensor - adjacency_tensor) < 1e-5

        logging.info('constructor_2 complete')

    def test_constructor_adjacency_tensor_1(self):
        """
        Test construction from adjacency_tensor.

        Validation of 3 numerical representations and other properties
        """
        logging.info('constructor_3 start')
        edge_list= [
            [[0], [1, 2]],
            [[0], [1, 3]],
            [[1], [0, 2]],
            [[1], [0, 3]],
            [[2], [0, 1]],
            [[3], [0, 1]]
        ]
        node_df = pd.DataFrame({'Nodes': [0,1,2,3]})
        data = {
            'Nodes': [[0, 1, 2], [0, 1, 3], [0, 1, 2], [0, 1, 3], [0, 1, 2], [0, 1, 3]],
            'Head': [[0], [0], [1], [1], [2], [3]],
            'Tail': [[1, 2], [1, 3], [0, 2], [0, 3], [0, 1], [0, 1]]
        }

        # Create the DataFrame with specified indices
        edges_df = pd.DataFrame(data, index=[0, 1, 4, 5, 8, 10])

        incidence_matrix = np.array(
        [[1,   1, -1, -1, -1, -1],
        [ -1, -1,  1,  1, -1, -1],
        [ -1,  0, -1,  0,  1,  0],
        [ 0,  -1,  0, -1,  0,  1]]
        )
        adjacency_tensor = np.array(
            [[[0, 0, 0, 0],
            [0, 0, 1, 1],
            [0, 1, 0, 0],
            [0, 1, 0, 0]],
            [[0, 0, 1, 1],
            [0, 0, 0, 0],
            [1, 0, 0, 0],
            [1, 0, 0, 0]],
            [[0, 1, 0, 0],
            [1, 0, 0, 0],
            [0, 0, 0, 0],
            [0, 0, 0, 0]],
            [[0, 1, 0, 0],
            [1, 0, 0, 0],
            [0, 0, 0, 0],
            [0, 0, 0, 0]]]
        )
        HG = Hypergraph(adjacency_tensor=adjacency_tensor)

        # Validate hypergraph properties
        assert HG.order == 3
        assert HG.uniform == True
        assert HG.directed == True
        np.testing.assert_array_equal(HG.adjacency_tensor, adjacency_tensor)

        # Validate dataframes
        try:
            assert_frame_equal(node_df, HG.nodes, check_dtype=False)
        except AssertionError as e:
            logging.info(f"DataFrames are not equal: {e}")

        try:
            assert_frame_equal(edges_df, HG.edges, check_dtype=False)
        except AssertionError as e:
            logging.info(f"DataFrames are not equal: {e}")

        assert np.sum(HG.adjacency_tensor - adjacency_tensor) < 1e-5

        np.testing.assert_array_equal(HG.incidence_matrix, incidence_matrix)
        assert are_nested_lists_equivalent(HG.edge_list, edge_list) == True

        logging.info('constructor_3 complete')

class HypergraphConstructorTestCase2(unittest.TestCase):
    def test_add_node_1(self):
        incidence_matrix=np.array(
            [[1, 1],
            [1, 1],
            [1, 0],
            [0, 1]]
        )
        HG = Hypergraph(incidence_matrix=incidence_matrix)
        HG.add_node()

        node_df = pd.DataFrame({'Nodes': [0,1,2,3,4]})
        assert_frame_equal(node_df, HG.nodes, check_dtype=False)

    def test_add_node_2(self):
        incidence_matrix=np.array(
            [[1, 1],
            [1, 1],
            [1, 0],
            [0, 1]]
        )
        HG = Hypergraph(incidence_matrix=incidence_matrix)
        HG.nodes['key'] = ['a', 'b', 'c', 'd']
        HG.add_node(properties={'key':'e'})

        node_df = pd.DataFrame({'Nodes': [0,1,2,3,4],
                                'key':['a','b','c','d','e']})
        assert_frame_equal(node_df, HG.nodes, check_dtype=False)

    def test_add_node_2(self):
        incidence_matrix=np.array(
            [[1, 1],
            [1, 1],
            [1, 0],
            [0, 1]]
        )
        HG = Hypergraph(incidence_matrix=incidence_matrix)
        HG.nodes['key'] = ['a', 'b', 'c', 'd']
        HG.add_node(properties={'key':'e', 'value':'z'})

        node_df = pd.DataFrame({'Nodes': [0,1,2,3,4],
                                'key':['a','b','c','d','e'],
                                'value':[pd.NA, pd.NA, pd.NA, pd.NA, 'z']})
        assert_frame_equal(node_df, HG.nodes, check_dtype=False)

    def test_add_node_3(self):
        incidence_matrix=np.array(
            [[1, 1],
            [1, 1],
            [1, 0],
            [0, 1]]
        )
        HG = Hypergraph(incidence_matrix=incidence_matrix)
        HG.nodes['key'] = ['a', 'b', 'c', 'd']
        HG.nodes['value'] = ['w','x','y','z']
        HG.add_node(properties={'key':'e'})

        node_df = pd.DataFrame({'Nodes': [0,1,2,3,4],
                                'key':['a','b','c','d','e'],
                                'value':['w','x','y','z',pd.NA]})
        assert_frame_equal(node_df, HG.nodes, check_dtype=False)

    def test_add_edge_1(self):
        incidence_matrix = np.array(
            [[1, 1],
            [1, 1],
            [1, 0],
            [0, 1]]
        )
        HG = Hypergraph(incidence_matrix=incidence_matrix)

        # Add properties to nodes and edges
        HG.edges['weight'] = [1.5, 2.5]

        # Add an edge
        HG.add_edge(nodes=[2, 3], properties={'weight': 3.0})

        # Define expected edge DataFrame
        edge_df = pd.DataFrame({
            'Nodes': [[0, 1, 2], [0, 1, 3], [2, 3]],
            'weight': [1.5, 2.5, 3.0]
        })

        # Assert the edges match
        assert_frame_equal(edge_df, HG.edges, check_dtype=False)

    def test_add_edge_2(self):
        incidence_matrix = np.array(
            [[1, 1],
            [1, 1],
            [1, 0],
            [0, 1]]
        )
        HG = Hypergraph(incidence_matrix=incidence_matrix)

        # Add properties to nodes and edges
        HG.edges['weight'] = [1.5, 2.5]

        # Add an edge
        HG.add_edge(nodes=[2, 3])

        # Define expected edge DataFrame
        edge_df = pd.DataFrame({
            'Nodes': [[0, 1, 2], [0, 1, 3], [2, 3]],
            'weight': [1.5, 2.5, pd.NA]
        })

        # Assert the edges match
        assert_frame_equal(edge_df, HG.edges, check_dtype=False)

    def test_add_edge_3(self):
        incidence_matrix = np.array(
            [[1, 1],
            [1, 1],
            [1, 0],
            [0, 1]]
        )
        HG = Hypergraph(incidence_matrix=incidence_matrix)

        # Add an edge
        HG.add_edge(nodes=[2, 3], properties={'weight': 3.0})

        # Define expected edge DataFrame
        edge_df = pd.DataFrame({
            'Nodes': [[0, 1, 2], [0, 1, 3], [2, 3]],
            'weight': [pd.NA, pd.NA, 3.0]
        })

        # Assert the edges match
        assert_frame_equal(edge_df, HG.edges, check_dtype=False)

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s\t - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)

if __name__ == '__main__':
    unittest.main()

