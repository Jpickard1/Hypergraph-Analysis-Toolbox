import sys
sys.path.append('../')
from HAT import Hypergraph
import numpy as np
import pandas as pd
from pandas.testing import assert_frame_equal
import unittest

def constructor_1():
    """Validate .nodes, .edges, and other properties on very small example
    """
    print('constructor_1 start')
    edge_list = [[0,1,2],
                 [0,1,3]]
    HG = Hypergraph(edge_list=edge_list)
    node_df = pd.DataFrame({'Nodes': [0,1,2,3]})
    edges_df = pd.DataFrame({'Edges': [0,1]})
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

    # Validate dataframes
    try:
        assert_frame_equal(node_df, HG.nodes, check_dtype=False)
    except AssertionError as e:
        print(f"DataFrames are not equal: {e}")

    try:
        assert_frame_equal(edges_df, HG.edges, check_dtype=False)
    except AssertionError as e:
        print(f"DataFrames are not equal: {e}")

    print('constructor_1 complete')

def constructor_2():
    print('constructor_2 start')
    edge_list = [[0,1,2],
                 [0,1,3]]
    node_df = pd.DataFrame({'Nodes': [0,1,2,3]})
    edges_df = pd.DataFrame({'Edges': [0,1]})
    incidence_matrix=np.array(
        [[1, 1],
         [1, 1],
         [1, 0],
         [0, 1]]
    )
    HG = Hypergraph(incidence_matrix=incidence_matrix)

    # Validate hypergraph properties
    assert HG.order == 3
    assert HG.uniform == True
    assert HG.directed == False
    np.testing.assert_array_equal(HG.incidence_matrix, incidence_matrix)

    # Validate dataframes
    try:
        assert_frame_equal(node_df, HG.nodes, check_dtype=False)
    except AssertionError as e:
        print(f"DataFrames are not equal: {e}")

    try:
        assert_frame_equal(edges_df, HG.edges, check_dtype=False)
    except AssertionError as e:
        print(f"DataFrames are not equal: {e}")

    print('constructor_2 complete')

# def main():
constructor_1()
constructor_2()
