import unittest
from Interpretable_Optimization.models.utils_models.utils_functions import linear_dependency, gurobi_to_pyomo
import gurobipy as gp
from gurobipy import GRB
import numpy as np
from scipy.sparse import csr_matrix
import pyomo.environ as pyo


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


if __name__ == '__main__':
    unittest.main()
