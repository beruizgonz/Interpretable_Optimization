import numpy as np
import gurobipy as gp

def get_row_activities(model):
    """
    Compute and return the support, minimal activity, and maximal activity for each row in a Gurobi model.

    Parameters:
    model (gurobipy.Model): The Gurobi model containing the optimization problem.

    Returns:
    support, min_activity, max_activity.
    SUPP - support: Set of indices j where a_ij is non-zero.
    INF - min_activity: Infimum of the row activity calculated as the sum of a_ij*l_j (for a_ij > 0) and a_ij*u_j (for a_ij < 0).
    SUP - max_activity: Supremum of the row activity calculated as the sum of a_ij*u_j (for a_ij > 0) and a_ij*l_j (for a_ij < 0).
    """

    rows = model.getConstrs()
    cols = model.getVars()
    SUPP = []
    SUP = []
    INF = []

    for i, row in enumerate(rows):
        support = set()
        min_activity = 0
        max_activity = 0

        for j, var in enumerate(cols):
            a_ij = model.getCoeff(row, var)
            l_j = var.LB
            u_j = var.UB

            if a_ij != 0:
                support.add(j)

                if a_ij > 0:
                    min_activity += a_ij * l_j
                    max_activity += a_ij * u_j
                else:
                    min_activity += a_ij * u_j
                    max_activity += a_ij * l_j

        SUPP.append(support)
        INF.append(min_activity)
        SUP.append(max_activity)

    return SUPP, INF, SUP


def feedback_individual_constraints(model, feasibility_tolerance=1e-6, infinity=1e30):
    """
    Analyzes each constraint in a Gurobi model and categorizes them as valid, redundant, or infeasible.

    Parameters:
    model (gurobipy.Model): The Gurobi model containing the optimization problem.
    feasibility_tolerance (float, optional): Tolerance used to assess feasibility. Defaults to 1e-6.
    infinity (float, optional): Value representing infinity in the context of the model. Defaults to 1e30.

    Returns:
    np.array: A matrix with each row containing the constraint number and its feedback ('valid', 'redundant', 'infeasible').
    """

    # Copy the model
    model_copy = model.copy()

    # Transform all '>=' constraints to '<=' in the copied model
    for constr in model_copy.getConstrs():
        if constr.Sense == '>':
            constr.Sense = '<='
            constr.RHS = -constr.RHS
            for i, var in enumerate(model_copy.getVars()):
                coeff = model_copy.getCoeff(constr, var)
                model_copy.chgCoeff(constr, var, -coeff)

    model_copy.update()
    # Get row activities from the model
    SUPP, INF, SUP = get_row_activities(model_copy)

    feedback_matrix = []

    for i, constr in enumerate(model_copy.getConstrs()):
        # Extracting constraint details
        sense = constr.Sense
        rhs = constr.RHS

        # Determining the feedback
        if sense == '<':
            if rhs >= infinity:
                feedback = 'redundant'
            elif SUP[i] <= rhs + feasibility_tolerance:
                feedback = 'redundant'
            elif INF[i] > rhs + feasibility_tolerance:
                feedback = 'infeasible'
            else:
                feedback = 'valid'
        elif sense == '=':
            if INF[i] >= rhs - feasibility_tolerance and SUP[i] <= rhs + feasibility_tolerance:
                feedback = 'redundant'
            elif INF[i] > rhs + feasibility_tolerance or SUP[i] < rhs - feasibility_tolerance:
                feedback = 'infeasible'
            else:
                feedback = 'valid'
        else:
            raise ValueError("Unsupported constraint sense: {}".format(sense))

        feedback_matrix.append([i, feedback])

    return np.array(feedback_matrix)