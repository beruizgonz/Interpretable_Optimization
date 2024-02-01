# =============================================== Metrics for optimality ===============================================

def relative_change_in_objective(original_value, modified_value, epsilon=1.0):
    """
    Calculate the relative change in the objective value as a percentage.

    Parameters:
    original_value (float): The objective value of the original problem.
    modified_value (float): The objective value of the modified problem.
    epsilon (float): A small number to represent 100% variation when the original value is zero.

    Returns:
    float: The percentage change in the objective value. If the original value is zero,
           the change is calculated based on the epsilon value.
    """
    # Check if the original value is zero
    if original_value == 0:
        if modified_value == 0:
            return 0  # No change if both are zero
        else:
            # Calculate the percentage change using epsilon as the baseline
            percentage_change = (modified_value / epsilon) * 100
            return percentage_change

    # Calculate the percentage change for non-zero original value
    percentage_change = ((modified_value - original_value) / original_value) * 100
    return percentage_change


def shadow_prices_changes(original_model, modified_model, epsilon=1.0):
    """
    Calculate the relative changes in shadow prices for each constraint between the original
    and modified models, and the relative average change across all constraints. Also identifies
    constraints that are missing in the modified model.

    Parameters:
    original_model: The solved original LP model.
    modified_model: The solved modified LP model.
    epsilon (float): A small number to represent 100% variation when the original value is zero.

    Returns:
    tuple: (rcc, rac, mcm)
        - rcc (dict): Relative change per constraint.
        - rac (float): Relative average change across all constraints.
        - mcm (dict): Missing constraints in the modified model with their shadow prices in the original model.
    """

    original_shadow_prices = {constraint.ConstrName: constraint.Pi for constraint in original_model.getConstrs()}
    modified_shadow_prices = {constraint.ConstrName: constraint.Pi for constraint in modified_model.getConstrs()}

    rcc = {}  # Relative change per constraint
    total_change = 0
    count = 0
    mcm = {}  # Missing constraints in the modified model

    for constr_name, original_price in original_shadow_prices.items():
        if constr_name in modified_shadow_prices:
            modified_price = modified_shadow_prices[constr_name]
            if original_price == 0:
                change = (modified_price / epsilon) * 100 if modified_price != 0 else 0
            else:
                change = ((modified_price - original_price) / original_price) * 100

            rcc[constr_name] = change
            total_change += abs(change)
            count += 1
        else:
            # Constraint is missing in the modified model
            mcm[constr_name] = original_price

    rac = total_change / count if count > 0 else 0  # Relative average change

    return rcc, rac, mcm


def basis_stability_analysis(original_model, modified_model, epsilon=1.0):
    """
    Analyze the stability of the basis between the original and modified LP problems.

    Parameters:
    original_model: The solved original LP model.
    modified_model: The solved modified LP model.
    epsilon (float): A small number to represent 100% variation when the original value is zero.

    Returns:
    tuple: (rcbd, rabd, psb, wpsb)
        - rcbd (dict): Relative change per basic decision variable.
        - rabd (float): Relative average change on basic decisions that match.
        - psb (float): Percentage of decisions on the same basis for original and modified.
        - wpsb (float): Weighted percentage of decisions on the same basis.
    """

    def extract_basic_variables(model):
        return {var.VarName: var.X for var in model.getVars() if var.VBasis == 0}

    original_basis_vars = extract_basic_variables(original_model)
    modified_basis_vars = extract_basic_variables(modified_model)

    rcbd = {}  # Relative change per basic decision variable
    total_change = 0
    count_matched = 0
    count_same_basis = 0
    total_value_original = 0
    total_value_matched = 0

    for var_name, original_value in original_basis_vars.items():
        total_value_original += abs(original_value)
        if var_name in modified_basis_vars:
            modified_value = modified_basis_vars[var_name]
            if original_value == 0:
                change = (modified_value / epsilon) * 100 if modified_value != 0 else 0
            else:
                change = ((modified_value - original_value) / original_value) * 100

            rcbd[var_name] = change
            total_change += abs(change)
            count_matched += 1
            total_value_matched += abs(modified_value)

            if original_value != 0 and modified_value != 0:
                count_same_basis += 1

    # Relative average change
    rabd = total_change / count_matched if count_matched > 0 else 0

    # Percentage on same basis
    psb = (count_same_basis / len(original_basis_vars)) * 100

    # Weighted percentage on same basis
    wpsb = (total_value_matched / total_value_original) * 100 if total_value_original > 0 else 0

    return rcbd, rabd, psb, wpsb


