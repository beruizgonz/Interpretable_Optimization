import unittest
import numpy as np
from scipy.sparse import csr_matrix
from Interpretable_Optimization.models.utils_models.utils_functions import linear_dependency


class TestLinearDependency(unittest.TestCase):
    def test_linear_dependency_dense(self):
        # Test case for dense matrix
        A = np.array([[1, 2], [2, 4], [3, 3]])
        b = np.array([1, 2, 3])
        expected_dependent_rows, expected_has_linear_dependency = ([[1], [0], []], np.array([True, True, False]))
        dependent_rows, has_linear_dependency = linear_dependency(A, b)

        # Check if the dependent rows match
        self.assertEqual(dependent_rows, expected_dependent_rows)

        # Check if the boolean arrays indicating linear dependency match
        self.assertTrue(np.array_equal(has_linear_dependency, expected_has_linear_dependency))

    def test_linear_dependency_sparse(self):
        # Test case for sparse matrix
        A = csr_matrix([[1, 0], [0, 0], [0, 1]])
        b = np.array([1, 0, 1])
        expected_dependent_rows, expected_has_linear_dependency = ([[], [], []], np.array([False, False, False]))
        dependent_rows, has_linear_dependency = linear_dependency(A, b)

        # Check if the dependent rows match
        self.assertEqual(dependent_rows, expected_dependent_rows)

        # Check if the boolean arrays indicating linear dependency match
        self.assertTrue(np.array_equal(has_linear_dependency, expected_has_linear_dependency))


if __name__ == '__main__':
    unittest.main()
