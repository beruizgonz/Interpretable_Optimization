import gurobipy as gp
import random
import numpy as np


def create_original_model(n_variables, n_constraints):
    """
    Create a random linear programming (LP) model with the specified number of decision variables and constraints.

    Parameters:
    - n_variables (int): Number of decision variables.
    - n_constraints (int): Number of constraints.

    Returns:
    - model (gurobipy.Model): The created LP model.
    """

    # Create a Gurobi model
    model = gp.Model("Original_LP")

    try:
        # Define decision variables
        variables = {}
        for i in range(n_variables):
            variables[i] = model.addVar(name=f'x_{i}')

        # Set objective function with random coefficients
        c = [random.uniform(-10, 10) for _ in range(n_variables)]
        model.setObjective(gp.quicksum(c[i] * variables[i] for i in range(n_variables)), sense=gp.GRB.MINIMIZE)

        # Add random constraints
        A = np.random.rand(n_constraints, n_variables)
        b = np.random.rand(n_constraints)
        for i in range(n_constraints):
            model.addConstr(gp.quicksum(A[i, j] * variables[j] for j in range(n_variables)) <= b[i],
                            name=f'Constraint_{i}')

        # Update model
        model.update()

    except gp.GurobiError as e:
        print(f"Gurobi Error: {e}")
        model = None

    return model
