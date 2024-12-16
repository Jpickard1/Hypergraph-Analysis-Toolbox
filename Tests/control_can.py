import sys
sys.path.append('../')
from HAT import Hypergraph, dynamics
import numpy as np
import unittest
import logging

class HypergraphControlTests(unittest.TestCase):
    def test_bmatrix_1(self):
        """
        Test construction of control configuration matrix
        """
        incidence_matrix=np.array(
            [[1, 1],
            [1, 1],
            [1, 0],
            [0, 1]]
        )
        HG = Hypergraph(incidence_matrix=incidence_matrix)

        B1 = dynamics.b_matrix(HG, [0])
        correct_b_matrix_1 = np.array([[1], [0], [0], [0]])
        B2 = dynamics.b_matrix(HG, [0, 1])
        correct_b_matrix_2 = np.array([[1, 0], [0, 1], [0, 0], [0, 0]])
        B3 = dynamics.b_matrix(HG, [1, 0])
        correct_b_matrix_3 = np.array([[0, 1], [1, 0], [0, 0], [0, 0]])

        np.testing.assert_array_equal(B1, correct_b_matrix_1, "B1 matrix does not match the expected output.")
        np.testing.assert_array_equal(B2, correct_b_matrix_2, "B2 matrix does not match the expected output.")
        np.testing.assert_array_equal(B3, correct_b_matrix_3, "B3 matrix does not match the expected output.")

    def test_ctrbk(self):
        """
        Test construction of control configuration matrix
        """
        logging.info('test_constructor_edge_set_1')
        incidence_matrix=np.array(
            [[1, 1],
            [1, 1],
            [1, 0],
            [0, 1]]
        )
        HG = Hypergraph(incidence_matrix=incidence_matrix)
        C1 = dynamics.ctrbk(HG, [0,1,2,3])
        correct_C_matrix_1 = np.array(
            [
                [1, 0, 0, 0],
                [0, 1, 0, 0],
                [0, 0, 1, 0],
                [0, 0, 0, 1]
            ]
        )
        incidence_matrix=np.array(
            [[1],
            [1],
            [1],
            [1]]
        )
        HG = Hypergraph(incidence_matrix=incidence_matrix)
        C2 = dynamics.ctrbk(HG, [0,1,2])
        correct_C_matrix_2 = np.array(
            [[0., 0., 0., 1.],
            [0., 1., 0., 0.],
            [0., 0., 1., 0.],
            [1., 0., 0., 0.]]
        )
        np.testing.assert_array_equal(C1, correct_C_matrix_1, "C1 matrix does not match the expected output.")
        np.testing.assert_array_equal(C2, correct_C_matrix_2, "C2 matrix does not match the expected output.")

if __name__ == '__main__':
    unittest.main()

