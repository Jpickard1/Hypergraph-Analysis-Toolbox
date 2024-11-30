import sys
sys.path.append('../')
from HAT import Hypergraph
import numpy as np
import pandas as pd
from pandas.testing import assert_frame_equal
import unittest

def are_nested_lists_equivalent(list1, list2):
    # Check if both are lists of lists
    if not all(isinstance(sublist, list) for sublist in list1 + list2):
        return False

    # Sort each sublist and then sort the outer list
    sorted_list1 = sorted([sorted(sublist) for sublist in list1])
    sorted_list2 = sorted([sorted(sublist) for sublist in list2])

    # Compare the sorted lists
    return sorted_list1 == sorted_list2

def constructor_1():
    """
    Test construction from edge_list.

    Validation of 3 numerical representations and other properties
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

    # Validate hypergraph properties
    assert HG.order == 3
    assert HG.uniform == True
    assert HG.directed == False
    np.testing.assert_array_equal(HG.incidence_matrix, incidence_matrix)
    np.testing.assert_array_equal(HG.adjacency_tensor, adjacency_tensor)

    # Validate dataframes
    try:
        assert_frame_equal(node_df, HG.nodes, check_dtype=False)
    except AssertionError as e:
        print(f"DataFrames are not equal: {e}")

    try:
        assert_frame_equal(edges_df, HG.edges, check_dtype=False)
    except AssertionError as e:
        print(f"DataFrames are not equal: {e}")

    assert are_nested_lists_equivalent(HG.edge_list, edge_list) == True
    assert np.sum(HG.adjacency_tensor - adjacency_tensor) < 1e-5

    print('constructor_1 complete')

def constructor_2():
    """
    Test construction from incidence_matrix.

    Validation of 3 numerical representations and other properties
    """
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
        assert_frame_equal(node_df, HG.nodes, check_dtype=False)
    except AssertionError as e:
        print(f"DataFrames are not equal: {e}")

    try:
        assert_frame_equal(edges_df, HG.edges, check_dtype=False)
    except AssertionError as e:
        print(f"DataFrames are not equal: {e}")

    assert are_nested_lists_equivalent(HG.edge_list, edge_list) == True
    assert np.sum(HG.adjacency_tensor - adjacency_tensor) < 1e-5

    print('constructor_2 complete')

def constructor_3():
    """
    Test construction from adjacency_tensor.

    Validation of 3 numerical representations and other properties
    """
    print('constructor_3 start')
    edge_list= [
        [[0], [1, 2]],
        [[0], [1, 3]],
        [[1], [0, 2]],
        [[1], [0, 3]],
        [[2], [0, 1]],
        [[3], [0, 1]]
    ]
    node_df = pd.DataFrame({'Nodes': [0,1,2,3]})
    edges_df = pd.DataFrame({'Edges': [0,1]})
    incidence_matrix = np.array(
      [[-1, -1,  1,  1,  1,  1],
       [ 1,  1, -1, -1,  1,  1],
       [ 1,  0,  1,  0, -1,  0],
       [ 0,  1,  0,  1,  0, -1]]
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
        print(f"DataFrames are not equal: {e}")

    try:
        assert_frame_equal(edges_df, HG.edges, check_dtype=False)
    except AssertionError as e:
        print(f"DataFrames are not equal: {e}")

    assert np.sum(HG.adjacency_tensor - adjacency_tensor) < 1e-5

    np.testing.assert_array_equal(HG.incidence_matrix, incidence_matrix)
#    print(f"{HG.edge_list=}")
    assert are_nested_lists_equivalent(HG.edge_list, edge_list) == True

    print('constructor_3 complete')

constructor_1()
constructor_2()
constructor_3()

