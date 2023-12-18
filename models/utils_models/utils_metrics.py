
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



