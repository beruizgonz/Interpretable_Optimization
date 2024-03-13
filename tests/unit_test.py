import unittest
from Interpretable_Optimization.models.utils_models.utils_functions import linear_dependency, gurobi_to_pyomo
import gurobipy as gp
from gurobipy import GRB
import numpy as np
from scipy.sparse import csr_matrix
import pyomo.environ as pyo
from Interpretable_Optimization.models.utils_models.utils_presolve import get_row_activities_fast, bound_strengthening


class UtilsFunctionsTest(unittest.TestCase):
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

    def test_gurobi_to_pyomo_conversion(self):
        # creating a simple gurobi model
        gurobi_model = gp.Model("simple_lp")
        x = gurobi_model.addVar(lb=0, name="x")
        y = gurobi_model.addVar(lb=0, name="y")

        # Set objective
        gurobi_model.setObjective(3 * x + 4 * y, GRB.MAXIMIZE)

        # Add constraints
        gurobi_model.addConstr(x + 2 * y <= 14, "c0")
        gurobi_model.addConstr(3 * x - y >= 0, "c1")

        gurobi_model.update()
        pyomo_model = gurobi_to_pyomo(gurobi_model)

        # Check variable count
        assert len(pyomo_model.vars) == 2

        # Check objective sense
        assert pyomo_model.obj.sense == 1

        # Check objective coefficients
        assert pyomo_model.obj.expr.to_string() == "3.0*vars[x] + 4.0*vars[y]"

        # Check constraints
        assert len(pyomo_model.cons) == 2

        # Check variable bounds and names
        for v in pyomo_model.vars.values():
            assert v.lb == 0
            # Check for specific variable bounds if needed

    def test_activity_row_fast_function_1(self):
        """
        Tests get_row_activities_fast with a small matrix where all coefficients are positive and bounds are straightforward.
        """
        # Define a small matrix with all positive coefficients
        A = csr_matrix([[1, 2], [3, 4]])  # Sparse representation of the matrix
        lb = [0, 0]  # Lower bounds
        ub = [3, 3]  # Upper bounds

        # Call the function to test
        support, min_activity, max_activity = get_row_activities_fast(A, lb, ub)

        # Define the expected results based on the provided inputs
        expected_support = [{0, 1}, {0, 1}]
        expected_min_activity = [0, 0]
        expected_max_activity = [9, 21]

        # Compare the expected results with the actual results
        self.assertEqual(support, expected_support, "Support does not match expected results.")
        self.assertEqual(min_activity.tolist(), expected_min_activity, "Min activity does not match expected results.")
        self.assertEqual(max_activity.tolist(), expected_max_activity, "Max activity does not match expected results.")

    def test_activity_row_fast_function_2(self):
        """
        Tests get_row_activities_fast with a matrix containing both positive and negative coefficients and non-uniform bounds.
        """
        # Define a matrix with both positive and negative coefficients
        A = csr_matrix([[1, -2], [-3, 4], [0, 5]])  # Sparse representation of the matrix
        lb = [-1, 0]  # Lower bounds
        ub = [2, 3]  # Upper bounds

        # Call the function to test
        support, min_activity, max_activity = get_row_activities_fast(A, lb, ub)

        # Define the expected results based on the provided inputs
        expected_support = [{0, 1}, {0, 1}, {1}]
        expected_min_activity = [-1 * 1 + (-2) * 3, (-3) * (2) + 4 * 0, 0 * (-1) + 5 * 0]  # Calculating manually
        expected_max_activity = [1 * 2 + (-2) * 0, (-3) * (-1) + 4 * 3, 0 * 2 + 5 * 3]  # Calculating manually

        # Compare the expected results with the actual results
        self.assertEqual(support, expected_support, "Support does not match expected results.")
        self.assertEqual(min_activity.tolist(), expected_min_activity, "Min activity does not match expected results.")
        self.assertEqual(max_activity.tolist(), expected_max_activity, "Max activity does not match expected results.")

    def test_activity_row_fast_function_3(self):
        """
        Tests get_row_activities_fast with a matrix that includes a row with all zero coefficients.
        """
        # Define a matrix with a row of all zeros and other coefficients
        A = csr_matrix([[1, 2], [0, 0], [3, -4]])  # Sparse representation of the matrix
        lb = [0, 1]  # Lower bounds
        ub = [3, 2]  # Upper bounds

        # Call the function to test
        support, min_activity, max_activity = get_row_activities_fast(A, lb, ub)

        # Define the expected results based on the provided inputs
        expected_support = [{0, 1}, set(), {0, 1}]  # Note the empty set for the all-zero row
        expected_min_activity = [1 * 0 + 2 * 1, 0, 3 * 0 + (-4) * 2]  # Min activity with zeros contributing nothing
        expected_max_activity = [1 * 3 + 2 * 2, 0, 3 * 3 + (-4) * 1]  # Max activity with zeros contributing nothing

        # Compare the expected results with the actual results
        self.assertEqual(support, expected_support, "Support does not match expected results.")
        self.assertEqual(min_activity.tolist(), expected_min_activity, "Min activity does not match expected results.")
        self.assertEqual(max_activity.tolist(), expected_max_activity, "Max activity does not match expected results.")

    def test_activity_row_fast_function_4(self):
        """
        Tests get_row_activities_fast with a matrix and bounds including extreme values (-np.inf, np.inf) to handle unbounded variables.
        """

        # Define a matrix with coefficients and include a mix of bounded and unbounded variables
        A = csr_matrix([[1, -2], [3, 0], [-1, 4]])  # Sparse representation of the matrix
        lb = [-np.inf, 0]  # Lower bounds, first variable unbounded below
        ub = [np.inf, 3]  # Upper bounds, first variable unbounded above

        # Call the function to test
        support, min_activity, max_activity = get_row_activities_fast(A, lb, ub)

        # Define the expected results based on the provided inputs
        expected_support = [{0, 1}, {0}, {0, 1}]
        expected_min_activity = [1*(-np.inf) + -2*3, 3*(-np.inf)+0*0, -1*(np.inf)+4*0]
        expected_max_activity = [1 * np.inf + (-2) * 0, 3 * np.inf + 0*3, -1 *(-np.inf) + 4 * 3]

        # Compare the expected results with the actual results
        self.assertEqual(support, expected_support, "Support does not match expected results.")
        self.assertTrue(np.all(np.isinf(min_activity) & (min_activity < 0)),
                        "Min activity does not match expected results for -inf bounds.")
        self.assertTrue(np.all(np.isinf(max_activity) & (max_activity > 0)),
                        "Max activity does not match expected results for inf bounds.")

    def test_bound_strengthening_basic(self):
        A = np.array([[1, -1], [-1, 2]])
        b = np.array([2, 4])
        lb = np.array([0, 0])
        ub = np.array([3, 3])

        expected_lb = np.array([2, 2])
        expected_ub = np.array([3, 3])

        new_lb, new_ub = bound_strengthening(A, b, lb, ub)

        np.testing.assert_array_almost_equal(new_lb, expected_lb, err_msg="Lower bounds did not match expected.")
        np.testing.assert_array_almost_equal(new_ub, expected_ub, err_msg="Upper bounds did not match expected.")

    def test_bound_strengthening_no_tightening(self):
        A = np.array([[1, 2], [2, 1]])
        b = np.array([5, 5])
        lb = np.array([0, 0])
        ub = np.array([10, 10])

        expected_lb = lb  # No tightening expected
        expected_ub = ub  # No tightening expected

        new_lb, new_ub = bound_strengthening(A, b, lb, ub)

        np.testing.assert_array_equal(new_lb, expected_lb, err_msg="Lower bounds changed unexpectedly.")
        np.testing.assert_array_equal(new_ub, expected_ub, err_msg="Upper bounds changed unexpectedly.")


if __name__ == '__main__':
    unittest.main()
