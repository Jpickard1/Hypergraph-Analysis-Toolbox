import sys
sys.path.append('../')
from HAT import Hypergraph, laplacian
import numpy as np
import unittest
import logging

class Laplacians(unittest.TestCase):
    def test_bolla(self):
        incidence_matrix=np.array(
            [[1, 1],
            [1, 1],
            [1, 0],
            [0, 1]]
        )
        HG = Hypergraph(incidence_matrix=incidence_matrix)
        L1 = laplacian.laplacian_matrix(HG)
        L2 = laplacian.laplacian_matrix(HG, type='Bolla')
        L3 = laplacian.bolla_laplacian(HG)
#        L4 = HG.laplacian
#        L5 = HG.laplacian(type='Bolla')
        L1_correct = np.array([[ 1.33333333, -0.66666667, -0.33333333, -0.33333333],
                               [-0.66666667,  1.33333333, -0.33333333, -0.33333333],
                               [-0.33333333, -0.33333333,  0.66666667,  0.        ],
                               [-0.33333333, -0.33333333,  0.        ,  0.66666667]])

        np.testing.assert_allclose(L1, L1_correct, atol=1e-5, err_msg="L1 matrix does not match the expected output.")
        np.testing.assert_allclose(L2, L1_correct, atol=1e-5, err_msg="L2 matrix does not match the expected output.")
        np.testing.assert_allclose(L3, L1_correct, atol=1e-5, err_msg="L3 matrix does not match the expected output.")
#        np.testing.assert_allclose(L4, L1_correct, atol=1e-5, err_msg="L4 matrix does not match the expected output.")
#        np.testing.assert_allclose(L5, L1_correct, atol=1e-5, err_msg="L5 matrix does not match the expected output.")

if __name__ == '__main__':
    unittest.main()

