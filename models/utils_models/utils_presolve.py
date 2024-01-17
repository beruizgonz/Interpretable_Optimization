
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
