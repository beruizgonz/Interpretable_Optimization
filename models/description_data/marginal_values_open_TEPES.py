import os 
import gurobipy as gp
import json
import sys
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider

# PATH TO THE DATA
parent_path = os.path.dirname(os.getcwd())
root_interpretable_optimization = os.path.dirname(parent_path)

# IMPORT FILES THAT ARE IN THE ROOT DIRECTORY
sys.path.append(parent_path)
from  models.utils_models.utils_functions import *
from models.utils_models.standard_model import *
from plot_matrix import *
from description_open_TEPES import *
from models.real_objective import *


# PATH TO THE DATA
GAMS_path = os.path.join(parent_path, 'data/GAMS_library_modified')
real_data_path = os.path.join(parent_path, 'data/real_data')
open_tepes_9n = os.path.join(real_data_path, 'openTEPES_EAPP_2030_sc01_st1.mps')
bounds_path = os.path.join(project_root, 'data/bounds_trace')
dinam_model = os.path.join(GAMS_path, 'DINAM.mps')
# Go to the modification presolve files
results_folder = os.path.join(parent_path, 'results_new/global/sparse/real_problems') 
results_sparsification_open_tepes_9n = os.path.join(results_folder, 'sparsification/epsilon_sparsification_openTEPES_EAPP_2030_sc01_st1.json')
results_cols_open_tepes_9n = os.path.join(results_folder, 'epsilon_cols_abs/epsilon_cols_openTEPES_EAPP_2030_sc01_st1.json')
results_rows_open_tepes_9n = os.path.join(results_folder, 'epsilon_rows_abs/epsilon_rows_openTEPES_EAPP_2030_sc01_st1.json')
# save paths
save_path_constraints = os.path.join(parent_path, 'figures_new/marginal_values/real_problems/importance')
save_path_zero = os.path.join(parent_path, 'figures_new/marginal_values/real_problems/zero_values')

def read_model(file_model):
    """
    Read the model
    """
    model = gp.read(file_model)
    model = standard_form_e2(model)
    #model= set_real_objective(model)
    return model

def change_constraint(model, num_constraint, multiplier = 1e-6):
    """
    Change a constraint to see how change the dual associated value
    """
    model.setParam('OutputFlag', 0)
    model.optimize()
    print('Model optimized: ', model.objVal)
    # Get the dual associated value of num_constraint
    m_constraints = list(model.getAttr('Pi', model.getConstrs()))
    m_constraints = np.array(m_constraints)
    sorted_index = np.argsort(m_constraints)
    dual_value = m_constraints[sorted_index][0]
    print('Dual value: ', dual_value)
    # Change the constraint in the model 
    constr = model.getConstrs()[sorted_index[0]]
    constr.setAttr('RHS', constr.RHS * multiplier)
    # Multiply all the lhs of the constraints by the same value
    specific_constr = model.getConstrs()[sorted_index[0]]

# Retrieve the LHS of the constraint as a linear expression
    lhs_expr = model.getRow(specific_constr)

    # Iterate through the variables in the constraint and modify their coefficients
    for i in range(lhs_expr.size()):
        var = lhs_expr.getVar(i)  # Get the variable
        coeff = lhs_expr.getCoeff(i)  # Get the coefficient
        # Update the coefficient by multiplying it with the multiplier
        model.chgCoeff(specific_constr, var, coeff * multiplier)

    # Update the model to apply the changes
    model.update()  
    model.optimize()
    print('Model optimized: ', model.objVal)
    m_constraints_new = list(model.getAttr('Pi', model.getConstrs()))
    dual_value_new = m_constraints_new[sorted_index[0]]
    print('Dual value new: ', dual_value_new)

def normalize_variables(model, alpha_min=0.1, alpha_max=10):
    """
    Normalize the variables of a Gurobi model to the range [0, 1].
    """
    A, b, c, co, lb, ub, of_sense, cons_senses, variable_names = calculate_bounds_candidates_sparse(model, None, None)
    model.setParam('OutputFlag', 0)
    model.optimize()
    print('Model optimized: ', model.objVal)
    if model.Status != gp.GRB.OPTIMAL:
        raise ValueError("Model optimization failed or is not optimal.")

    optimal_vars = [var.X for var in model.getVars()]
    # print the min lb and max ub
    lb = np.array(lb)   
    ub = np.array(ub)

    for i, name in enumerate(variable_names):
        if np.isinf(ub[i]) or  lb[i] == ub[i]:
            var_optimal = optimal_vars[i]
            if var_optimal is not None:
                if var_optimal > 0:
                    lb[i] = var_optimal * alpha_min
                    ub[i] = var_optimal * alpha_max
                else:
                    lb[i] = var_optimal * alpha_max
                    ub[i] = var_optimal * alpha_min
                    
        if lb[i] == 0 and ub[i] == 0 or (ub[i]-lb[i]) <= 1e-6:
            lb[i] = 0
            ub[i] = alpha_max

    # Check that all the bounds are finite and valid
    if not np.all(np.isfinite(lb)) or not np.all(np.isfinite(ub)):
        raise ValueError("Invalid bounds detected.")
    constraint_names = [constr.ConstrName for constr in model.getConstrs()]
    model_bounds = build_model(A, b, c, co, lb, ub, of_sense, cons_senses, variable_names, constraint_names)
    normalized_model = gp.Model("normalized_model")
    scaling_factors = {}
    normalized_vars = {}
    original_to_normalized = {}

    for i, name in enumerate(variable_names):
        if not (np.isfinite(lb[i]) and np.isfinite(ub[i])):
            raise ValueError(f"Variable {name} has invalid bounds: LB={lb[i]}, UB={ub[i]}")
        if lb[i] >= ub[i]:
            print(optimal_vars[i])
            raise ValueError(f"Variable {name} has invalid bounds: LB={lb[i]} >= UB={ub[i]}")


    for i, var in enumerate(model_bounds.getVars()):
        lower_bound = lb[i]
        upper_bound = ub[i]
        scale = upper_bound - lower_bound
        if scale == 'inf' or scale == '-inf' or scale == 0:
            raise ValueError(f"Variable {var.VarName} has invalid bounds: LB={lower_bound}, UB={upper_bound}.")

        scaling_factors[var] = (lower_bound, scale)
        norm_var = normalized_model.addVar(lb=0, ub=1, name=f"{var.VarName}")
        normalized_vars[var] = norm_var
        original_to_normalized[var.VarName] = norm_var

    normalized_model.update()

    old_obj = model_bounds.getObjective()
    new_obj = gp.LinExpr()

    for i in range(old_obj.size()):
        var = old_obj.getVar(i)
        if var not in scaling_factors:
            raise ValueError(f"Variable {var.VarName} not found in scaling_factors.")
        coeff = old_obj.getCoeff(i)
        lower_bound, scale = scaling_factors[var]
        new_obj += coeff * (scale * normalized_vars[var] + lower_bound)

    if model_bounds.ModelSense == gp.GRB.MINIMIZE:
        normalized_model.setObjective(new_obj, gp.GRB.MINIMIZE)
    else:
        normalized_model.setObjective(new_obj, gp.GRB.MAXIMIZE)

    # # Step 6: Add normalized constraints
    for constr in model_bounds.getConstrs():
        lhs = model_bounds.getRow(constr)
        rhs = constr.RHS
        sense = constr.Sense

        new_lhs = gp.LinExpr()
        for j in range(lhs.size()):
            var = lhs.getVar(j)
            coeff = lhs.getCoeff(j)
            lower_bound, scale = scaling_factors[var]
            new_lhs += coeff * (scale * normalized_vars[var] + lower_bound)

        # Add the constraint to the normalized model
        if sense == gp.GRB.LESS_EQUAL:
            normalized_model.addConstr(new_lhs <= rhs, name=constr.ConstrName)
        elif sense == gp.GRB.GREATER_EQUAL:
            normalized_model.addConstr(new_lhs >= rhs, name=constr.ConstrName)
        elif sense == gp.GRB.EQUAL:
            normalized_model.addConstr(new_lhs == rhs, name=constr.ConstrName)

    normalized_model.update()
    return normalized_model, scaling_factors

def set_bounds(model, alpha_min=0.1, alpha_max=10):
    A, b, c, co, lb, ub, of_sense, cons_senses, variable_names = calculate_bounds_candidates_sparse(model, None, None)
    model.setParam('OutputFlag', 0)
    model.optimize()
    print('Model optimized: ', model.objVal)
    if model.Status != gp.GRB.OPTIMAL:
        raise ValueError("Model optimization failed or is not optimal.")

    optimal_vars = [var.X for var in model.getVars()]
    # print the min lb and max ub
    lb = np.array(lb)   
    ub = np.array(ub)

    for i, name in enumerate(variable_names):
        if np.isinf(ub[i]) or  lb[i] == ub[i]:
            var_optimal = optimal_vars[i]
            if var_optimal is not None:
                if var_optimal > 0:
                    lb[i] = var_optimal * alpha_min
                    ub[i] = var_optimal * alpha_max
                else:
                    lb[i] = var_optimal * alpha_max
                    ub[i] = var_optimal * alpha_min
                    
        if lb[i] == 0 and ub[i] == 0 or (ub[i]-lb[i]) <= 1e-6:
            lb[i] = 0
            ub[i] = alpha_max
    constraint_names = [constr.ConstrName for constr in model.getConstrs()]
    model_bound = build_model(A, b, c, co, lb, ub, of_sense, cons_senses, variable_names, constraint_names) 
    return model_bound

def set_bounds_optimal(model, alpha_min=0.1, alpha_max=10):
    A, b, c, co, lb, ub, of_sense, cons_senses, variable_names = get_model_matrices(model)
    model.setParam('OutputFlag', 0)
    model.optimize()

    
    optimal_vars = [var.X for var in model.getVars()]
    # print the min lb and max ub
    lb = np.array(lb)   
    ub = np.array(ub)

    for i, name in enumerate(variable_names):
        if np.isinf(ub[i]) or  lb[i] == ub[i]:
            var_optimal = optimal_vars[i]
            if var_optimal is not None:
                if var_optimal > 0:
                    lb[i] = var_optimal * alpha_min
                    ub[i] = var_optimal * alpha_max
                else:
                    lb[i] = var_optimal * alpha_max
                    ub[i] = var_optimal * alpha_min
                    
        if lb[i] == 0 and ub[i] == 0 or (ub[i]-lb[i]) <= 1e-6:
            lb[i] = 0
            ub[i] = alpha_max
    constraint_names = [constr.ConstrName for constr in model.getConstrs()]
    model_bound = build_model(A, b, c, co, lb, ub, of_sense, cons_senses, variable_names, constraint_names)
    return model_bound


def get_marginal_values(model):
    """
    Parameters:
    - model: Gurobi model object.
    Returns:
    - objective_value: Objective value of the model.
    - marginal_variables: List of reduced costs (marginal values) corresponding to each variable in the model.
    - marginal_constraints: List of shadow prices (marginal values) corresponding to each constraint in the model.
    """
    model_norm = model
    model_norm.setParam('OutputFlag', 0)
    model_norm.optimize()
    A, b, c, co, lb, ub, of_sense, cons_senses, variable_names, constraint_names = get_model_matrices(model_norm)
    print('Model optimized: ', model_norm.objVal)
    marginal_variables = list(model_norm.getAttr('RC', model_norm.getVars()))
    marginal_constraints = list(model_norm.getAttr('Pi', model_norm.getConstrs()))

    # Divide the m_constraints by the constraint range  
    marginal_constraints = np.array(marginal_constraints)
    variables = [var.X for var in model_norm.getVars()]
    coeff = np.array([var.Obj for var in model.getVars()])
    c = np.array(c)
    importance = np.abs(c - A.T.dot(marginal_constraints).flatten())
    return model_norm.ObjVal, variables, marginal_variables, marginal_constraints, importance

def constraints_importance(model): 
    model.setParam('OutputFlag',0)
    model.optimize()
    marginal_constraints = list(model.getAttr('Pi', model.getConstrs()))
    # For all constraints obtain the value of the solution
    A, b, c, co, lb, ub, of_sense, cons_senses, variable_names, constraint_names = get_model_matrices(model)
    constrs_names = [constr.ConstrName for constr in model.getConstrs()]
    importance = np.array(marginal_constraints)
    # divide by the sum of the row of A
    A = csr_matrix(A)
    A = abs(A)
    row_sum = np.array(A.sum(axis=1)).flatten()
    importance = importance / row_sum   
    return importance, constraint_names

def variables_importance(model):
    model.setParam('OutputFlag',0)
    model.optimize()
    marginal_variables = list(model.getAttr('RC', model.getVars()))
    marginal_constraints = list(model.getAttr('Pi', model.getConstrs()))
    solution = [var.X for var in model.getVars()]
    A, b, c, co, lb, ub, of_sense, cons_senses, variable_names, constraint_names = get_model_matrices(model)
    constraints_values = A.dot(solution)
    vars_names = [var.VarName for var in model.getVars()]
    c = np.array(c)
    solution = np.array(solution)
    importance = np.abs(c - A.T.dot(marginal_constraints).flatten())
    return importance, vars_names

def importances_by_groups(importances, names, model, metric, group):
    if group == 'constraints': 
        names, asos, groups, inverted_group = groups_by_constraints(model)
    elif group == 'variables':
        names, asos, groups, inverted_group = groups_by_variables(model)
    total_importance = dict.fromkeys(groups.keys(), 0)
    for index, name in enumerate(names):
        group_name = inverted_group[name]
        total_importance[group_name] += np.abs(importances[index])
    if metric == 'sum':
        total_importance = {k: v for k, v in total_importance.items()}
    elif metric == 'mean':
        total_importance = {k: v / len(groups[k]) for k, v in total_importance.items()}
    total_importance = dict(sorted(total_importance.items(), key=lambda item: item[1], reverse=True))
    return total_importance

def plot_importances_values(model,importances, names, save_path, group): 
    if group == 'constraints': 
        names, asos, groups, inverted_group = groups_by_constraints(model)
    elif group == 'variables':
        names, asos, groups, inverted_group = groups_by_variables(model)
    total_importance = dict.fromkeys(groups.keys(), 0)
    for index, name in enumerate(names):
        group_name = inverted_group[name]
        total_importance[group_name] += np.abs(importances[index])
    # Order the importance by the total importance
    total_importance = dict(sorted(total_importance.items(), key=lambda item: item[1], reverse=True))
    # Normalize total importance so that the sum is 1
    total_importance = {k: v / sum(total_importance.values()) for k, v in total_importance.items()}
    plt.figure(figsize=(10, 5))
    plt.bar(range(len(total_importance)), total_importance.values(), color='skyblue', edgecolor='black')
    plt.xticks(range(len(total_importance)), total_importance.keys(), rotation=90)
    plt.xlabel(f'{group} groups')
    plt.ylabel('Importance values')
    plt.title(f'Importance for {group}: openTEPES')
    plt.tight_layout()
    names = f'importance_{group}_openTEPES.png' 
    plt.savefig(os.path.join(save_path, names))
    plt.show()
    
    
def zero_importance_values(model):    
    model.setParam('OutputFlag', 0)
    model.optimize()
    marginal_constraints = list(model.getAttr('Pi', model.getConstrs()))
    solution = [var.X for var in model.getVars()]
    A, b, c, co, lb, ub, of_sense, cons_senses, variable_names, constraint_names = get_model_matrices(model)
    c = np.array(c)
    constraints_values = A.dot(solution)
    vars_names = [var.VarName for var in model.getVars()]
    constrs_names = [constr.ConstrName for constr in model.getConstrs()]
    importance_constraints =np.abs(marginal_constraints)

    importance_vars = np.abs(c - A.T.dot(marginal_constraints).flatten())
    _, asos_constraints, group_contrs, inverted_group_constrs = groups_by_constraints(model)
    _, asos_vars, group_vars, inverted_group_vars = groups_by_variables(model)
    group_zero_constraints = dict.fromkeys(group_contrs.keys(), 0)
    group_zero_variables = dict.fromkeys(group_vars.keys(), 0)
    for index, name in enumerate(constrs_names):
        if importance_constraints[index] == 0:
            group_name = inverted_group_constrs[name]
            if group_name in group_zero_constraints:
                group_zero_constraints[group_name] += 1
    for index, name in enumerate(vars_names):
        if importance_vars[index] == 0:
            group_name = inverted_group_vars[name]
            if group_name in group_zero_variables:
                group_zero_variables[group_name] += 1
    total_vars_group = get_total_by_group(asos_vars)
    total_constr_group = get_total_by_group(asos_constraints)
    return group_zero_constraints, group_zero_variables, total_vars_group, total_constr_group

def plot_zero_importance(group_zero_constraints, group_zero_variables, total_vars, total_constrs, save_path):
    plt.figure(figsize=(10, 5))
    group_zero_constraints = dict(sorted(group_zero_constraints.items(), key=lambda item: item[1], reverse=True))
    total_constrs = dict(sorted(total_constrs.items(), key=lambda item: item[1], reverse=True))
    groups = list(total_constrs.keys())
    
    bar_width = 0.4
    indices = range(len(groups))
    
    plt.bar(indices, total_constrs.values(), width=bar_width, color='lightcoral', edgecolor='black', alpha=0.7, label='Total constraints')
    plt.bar([i + bar_width for i in indices], [group_zero_constraints.get(g, 0) for g in groups], width=bar_width, color='skyblue', edgecolor='black', alpha=0.7, label='Zero importance constraints')
    
    plt.xticks([i + bar_width / 2 for i in indices], groups, rotation=90)
    plt.xlabel('Constraints groups')
    plt.ylabel('Number of constraints')
    plt.title('Comparison of Total and Zero Importance Constraints per Group')
    plt.legend()
    plt.tight_layout()
    name_constrs = 'group_zero_constraints_comparison.png'
    plt.savefig(os.path.join(save_path, name_constrs))
    plt.show()

    plt.figure(figsize=(10, 5))
    group_zero_variables = dict(sorted(group_zero_variables.items(), key=lambda item: item[1], reverse=True))
    total_vars = dict(sorted(total_vars.items(), key=lambda item: item[1], reverse=True))
    groups_vars = list(total_vars.keys())
    
    indices_vars = range(len(groups_vars))
    
    plt.bar(indices_vars, total_vars.values(), width=bar_width, color='lightcoral', edgecolor='black', alpha=0.7, label='Total variables')
    plt.bar([i + bar_width for i in indices_vars], [group_zero_variables.get(g, 0) for g in groups_vars], width=bar_width, color='skyblue', edgecolor='black', alpha=0.7, label='Zero importance variables')
    
    plt.xticks([i + bar_width / 2 for i in indices_vars], groups_vars, rotation=90)
    plt.xlabel('Variables groups')
    plt.ylabel('Number of variables')
    plt.title('Comparison of Total and Zero Importance Variables per Group')
    plt.legend()
    plt.tight_layout()
    name_vars = 'group_zero_variables_comparison.png'
    plt.savefig(os.path.join(save_path, name_vars)) 
    plt.show()

def normalize_constraints_sparse(A, b):
    """
    Normalize the constraints of a Gurobi model to the range [0, 1].

    Args:
        A: 2D sparse matrix (constraints matrix, CSR format preferred)
        b: 1D or 2D array (right-hand side values)

    Returns:
        A: Normalized sparse matrix (CSR format)
        b: Normalized right-hand side vector
        scalers: Original values of b to reverse normalization if needed
    """
    # Ensure A is a sparse matrix
    A = csr_matrix(A)
    
    # Ensure b is a NumPy array and reshape it to a column vector
    b = np.array(b).reshape(-1, 1)

    # Save the original b values for scaling back the objective later
    scalers = np.where(b == 0, 1, b)  # Replace zeros in b with 1 to avoid division by zero

    # Get the nonzero indices and data of A
    row_indices, col_indices = A.nonzero()  # Get nonzero indices of A
    A_data = A.data  # Access nonzero elements of A

    # Safely normalize rows of A
    mask = b[row_indices].flatten() != 0  # Check where b[row_indices] is not zero
    A_normalized_data = np.zeros_like(A_data)  # Initialize normalized data array
    A_normalized_data[mask] = A_data[mask] / b[row_indices[mask]].flatten()  # Only divide where b != 0

    # Update the sparse matrix with normalized data
    A = csr_matrix((A_normalized_data, (row_indices, col_indices)), shape=A.shape)

    # Normalize b (setting entries corresponding to b == 0 to 0 to avoid invalid values)
    b_normalized = np.where(b != 0, b / scalers, 0)

    return A, b_normalized.flatten(), scalers


def normalize_constraints_bounds(model):
    """
    Normalize the constraints using the bounds of the model.
    Keeps the constraint matrix A in sparse format.
    
    Args:
        model: Optimization model to extract matrices and bounds.
    
    Returns:
        A: Normalized sparse constraint matrix (CSR format).
        b: Normalized right-hand side vector.
        constraint_range: Array of constraint ranges for each row.
    """
    # Get the model matrices and bounds
    A, b, c, co, lb, ub, of_sense, cons_senses, variable_names, constraint_names = get_model_matrices(model)
    
    # Ensure bounds are numpy arrays
    ub = np.array(ub)
    lb = np.array(lb)

    # Ensure A is in CSR format
    if not isinstance(A, csr_matrix):
        A = A.tocsr()

    # Get sparse matrix components
    row_indices, col_indices = A.nonzero()  # Get row and column indices of nonzero elements
    A_data = A.data  # Nonzero values in the sparse matrix

    # Initialize arrays to store maximum and minimum constraints for each row
    row_max_constraints = np.zeros(A.shape[0])  # One value per row
    row_min_constraints = np.zeros(A.shape[0])  # One value per row

    # Iterate over the rows of the sparse matrix
    for row in range(A.shape[0]):
        # Get indices of nonzero elements in the current row
        start_idx = A.indptr[row]
        end_idx = A.indptr[row + 1]

        # Extract row data
        row_data = A_data[start_idx:end_idx]
        row_cols = col_indices[start_idx:end_idx]

        # Compute maximum and minimum constraints for the row
        row_max = np.sum(np.maximum(row_data * ub[row_cols], row_data * lb[row_cols]))
        row_min = np.sum(np.minimum(row_data * ub[row_cols], row_data * lb[row_cols]))
        # row_max = A[row].dot(ub)
        # row_min = A[row].dot(lb)
        # Store the results
        if row_max == np.nan or row_min == np.nan:
            print('Row max: ', row_max)
            print('Row min: ', row_min)
        row_max_constraints[row] = row_max
        row_min_constraints[row] = row_min
    # Compute the range for each row
    constraint_range = abs(row_max_constraints - row_min_constraints)
    constraint_range[constraint_range == 0] = 1  # Avoid division by zero

    # Normalize A row-wise using the range
    A_data_normalized = A_data / constraint_range[row_indices]
    A = csr_matrix((A_data_normalized, (row_indices, col_indices)), shape=A.shape)
    # Normalize the right-hand side vector b
    b = np.array(b) -row_min / constraint_range
    return A, b, constraint_range


def marginal_values_open_tepes(model, group = True):
    """
    Calculate the marginal values for the openTEPES model
    """
    obj_value, variables, marginal_variables, marginal_constraints, importance = get_marginal_values(model)

    if group: 
        names_vars, asos_vars, dict_vars, inverted_group_v = groups_by_variables(model)
        names_constrs, asos_constr, dict_constr, inverted_group_c = groups_by_constraints(model) 
        total_marginal_values_variables = dict.fromkeys(dict_vars.keys(), 0)
        total_marginal_values_constraints = dict.fromkeys(dict_constr.keys(), 0)
        total_importance_variables = dict.fromkeys(dict_vars.keys(), 0)
        postoname = map_position_to_names(model)
        for position, names in postoname.items(): 
            index_var = position[1]
            index_constr = position[0]
            name_var = names[1]
            name_constr = names[0]
            if inverted_group_v[name_var].startswith('slack'):
                name_var_g = 'slack'
            else:
                name_var_g = inverted_group_v[name_var]
            name_constr_g = inverted_group_c[name_constr]
            total_marginal_values_variables[name_var_g] += abs(marginal_variables[index_var])
            total_importance_variables[name_var_g] += importance[index_var]
            total_marginal_values_constraints[name_constr_g] +=abs(marginal_constraints[index_constr])
        # Order the groups by the total marginal value
        total_marginal_values_variables = dict(sorted(total_marginal_values_variables.items(), key=lambda item: item[1], reverse=True))
        total_marginal_values_constraints = dict(sorted(total_marginal_values_constraints.items(), key=lambda item: item[1], reverse=True))
        # Normalize the values
        total_marginal_values_variables = {k: v / sum(total_marginal_values_variables.values()) for k, v in total_marginal_values_variables.items()}
        total_marginal_values_constraints = {k: v / sum(total_marginal_values_constraints.values()) for k, v in total_marginal_values_constraints.items()}
        plt.figure(figsize=(10, 5))
        plt.bar(range(len(total_marginal_values_variables)), total_marginal_values_variables.values(), color='skyblue', edgecolor='black')
        plt.xticks(range(len(total_marginal_values_variables)), total_marginal_values_variables.keys(), rotation=90)
        plt.xlabel('Variables groups')
        plt.ylabel('Marginal values')
        plt.title('Marginal values for Variables: openTEPES')
        plt.tight_layout()
        plt.savefig('marginal_variables_openTEPES.png')
        plt.show()
     
        # Make the same for the constraints
        plt.figure(figsize=(10, 5))
        plt.bar(range(len(total_marginal_values_constraints)), total_marginal_values_constraints.values(), color='skyblue', edgecolor='black')
        plt.xticks(range(len(total_marginal_values_constraints)), total_marginal_values_constraints.keys(), rotation=90)
        plt.xlabel('Constraints groups')
        plt.ylabel('Marginal values')
        plt.title('Marginal values for Constraints: openTEPES')
        plt.tight_layout()
        plt.savefig('marginal_constraints_openTEPES.png')
        plt.show()
       
        # Order the groups by the total importance
        total_importance_variables = dict(sorted(total_importance_variables.items(), key=lambda item: item[1], reverse=True))
        plt.figure(figsize=(10, 5))
        plt.bar(range(len(total_importance_variables)), total_importance_variables.values(), color='skyblue', edgecolor='black')
        plt.xticks(range(len(total_importance_variables)), total_importance_variables.keys(), rotation=90)
        plt.xlabel('Variables groups')
        plt.ylabel('Importance values')
        plt.title('Importance for Variables: openTEPES')
        plt.tight_layout()
        plt.savefig('importance_variables_openTEPES.png')
        plt.show()

def marginal_by_groups(model):
    obj_value, variables, marginal_variables, marginal_constraints, importance = get_marginal_values(model)
    names_vars, asos_vars, dict_vars, inverted_group_v = groups_by_variables(model)
    names_constrs, asos_constr, dict_constr, inverted_group_c = groups_by_constraints(model) 
    total_marginal_values_variables = dict.fromkeys(dict_vars.keys(), 0)
    total_marginal_values_constraints = dict.fromkeys(dict_constr.keys(), 0)
    total_importance_variables = dict.fromkeys(dict_vars.keys(), 0)
    postoname = map_position_to_names(model)
    for position, names in postoname.items(): 
        index_var = position[1]
        index_constr = position[0]
        name_var = names[1]
        name_constr = names[0]
        if inverted_group_v[name_var].startswith('slack'):
            name_var_g = 'slack'
        else:
            name_var_g = inverted_group_v[name_var]
        name_constr_g = inverted_group_c[name_constr]
        total_marginal_values_variables[name_var_g] += abs(marginal_variables[index_var])
        total_importance_variables[name_var_g] += importance[index_var]
        total_marginal_values_constraints[name_constr_g] +=abs(marginal_constraints[index_constr])
    # Get the total marginal values for the variables and constraints by the number of elements
    total_marginal_values_variables = {k: v / len(dict_vars[k]) for k, v in total_marginal_values_variables.items()}
    total_marginal_values_constraints = {k: v / len(dict_constr[k]) for k, v in total_marginal_values_constraints.items()}
    return total_marginal_values_variables, total_marginal_values_constraints, total_importance_variables 

def plot_marginal_values(total_marginal_values_variables, total_marginal_values_constraints, total_importance_variables):
    plt.figure(figsize=(10, 5))
    plt.bar(range(len(total_marginal_values_variables)), total_marginal_values_variables.values(), color='skyblue', edgecolor='black')
    plt.xticks(range(len(total_marginal_values_variables)), total_marginal_values_variables.keys(), rotation=90)
    plt.xlabel('Variables groups')
    plt.ylabel('Marginal values')
    plt.title('Marginal values for Variables: openTEPES')
    plt.tight_layout()
    plt.savefig('marginal_variables_openTEPES.png')
    plt.show()
    
    # Make the same for the constraints
    plt.figure(figsize=(10, 5))
    plt.bar(range(len(total_marginal_values_constraints)), total_marginal_values_constraints.values(), color='skyblue', edgecolor='black')
    plt.xticks(range(len(total_marginal_values_constraints)), total_marginal_values_constraints.keys(), rotation=90)
    plt.xlabel('Constraints groups')
    plt.ylabel('Marginal values')
    plt.title('Marginal values for Constraints: openTEPES')
    plt.tight_layout()
    plt.savefig('marginal_constraints_openTEPES.png')
    plt.show()
    
    # Order the groups by the total importance
    total_importance_variables = dict(sorted(total_importance_variables.items(), key=lambda item: item[1], reverse=True))
    plt.figure(figsize=(10, 5))
    plt.bar(range(len(total_importance_variables)), total_importance_variables.values(), color='skyblue', edgecolor='black')
    plt.xticks(range(len(total_importance_variables)), total_importance_variables.keys(), rotation=90)
    plt.xlabel('Variables groups')
    plt.ylabel('Importance values')
    plt.title('Importance for Variables: openTEPES')
    plt.tight_layout()
    plt.savefig('importance_variables_openTEPES.png')
    plt.show()
     
def importance_constraints(dual_model):
    dual_model_norm = normalize_variables(dual_model)
    dual_model_norm.setParam('OutputFlag', 0)
    dual_model_norm.optimize()
    print('Dual model optimized: ', dual_model_norm.objVal)
    m_constraints = [var.X for var in dual_model_norm.getVars()]
    names_vars, asos_vars, dict_vars, inverted_group_v = groups_by_variables(dual_model)
    names_constrs, asos_constr, dict_constr, inverted_group_c = groups_by_constraints(dual_model)
    total_importance_constraints = dict.fromkeys(dict_vars.keys(), 0)
    postoname = map_position_to_names(dual_model)
    for position, names in postoname.items(): 
        index_constr = position[1]
        name_constr = names[1]
        name_constr_g = inverted_group_v[name_constr]
        total_importance_constraints[name_constr_g] += abs(m_constraints[index_constr])
    print(total_importance_constraints)
    # Order the importance by the total importance
    total_importance_constraints = dict(sorted(total_importance_constraints.items(), key=lambda item: item[1], reverse=True))
    total_relative_importance = {k: v * len(dict_vars[k]) / sum(total_importance_constraints.values()) for k, v in total_importance_constraints.items()}
    plt.figure(figsize=(10, 5))
    plt.bar(range(len(total_relative_importance)), total_importance_constraints.values(), color='skyblue', edgecolor='black')
    plt.xticks(range(len(total_importance_constraints)), total_relative_importance.keys(), rotation=90)
    plt.xlabel('Constraints groups')
    plt.ylabel('Marginal values')
    plt.title('Marginal values for Constraints: openTEPES')
    plt.tight_layout()
    plt.savefig('marginal_constraints_openTEPES_dual.png')
    plt.show()

def multiple_solutions(model):
    model.setParam('OutputFlag', 0)
    model.optimize()
    solution = [var.X for var in model.getVars()]
    # Add a constraint to the model
    obj_eq = model.getObjective()
    model.addConstr(obj_eq == model.ObjVal)
    # Add a constraint to the model
    variable_0 = solution[0]
    model.addConstr(model.getVars()[0] >= variable_0 + 1e-4)
    model.update()
    model.optimize()
    print(model.objVal)

def dual_values(model):
    dual = construct_dual_model_sparse(model)
    dual.setParam('OutputFlag', 0)
    dual.optimize()
    print(dual.objVal)
    dual_solution = [var.X for var in dual.getVars()]
    dual_name = [var.VarName for var in dual.getVars()]
    A, b, c, co, lb, ub, of_sense, cons_senses, variable_names, constraint_names = get_model_matrices(model)
    A = csr_matrix(A)
    A = abs(A)
    row_sum = np.array(A.sum(axis=1)).flatten()
    dual_solution = dual_solution / row_sum
    return dual_solution, dual_name

def check_dual_solution(model):
    dual_model = construct_dual_model_sparse(model)
    dual_model.setParam('OutputFlag', 0)
    dual_model.optimize()
    # print the dual obj value
    print(dual_model.objVal)
    dual_solution = [var.X for var in dual_model.getVars()]
    dual_name = [var.VarName for var in dual_model.getVars()]
    
    model.setParam('OutputFlag', 0)
    model.optimize()

    print(model.SolCount)
    shadow_prices = list(model.getAttr('Pi', model.getConstrs()))

    # # Check the dual solution is the same as the shadow prices
    shadow_prices = np.array(shadow_prices)
    dual_solution = np.array(dual_solution)
    print(np.allclose(shadow_prices, dual_solution, atol=1e-6))

if __name__ == '__main__':

    model_open_tepes = read_model(open_tepes_9n)
    A, b, c, co, lb, ub, of_sense, cons_senses, var_names, constraint_names = get_model_matrices(model_open_tepes) 
 
    model_open_tepes_1 = set_bounds(model_open_tepes)
    model_open_tepes_1.setParam('OutputFlag', 0)  
    model_open_tepes_1.optimize()
    solution = [var.X for var in model_open_tepes_1.getVars()]
    solution = np.array(solution)
    print(solution[0:10])
    # A, b, c, co, lb, ub, of_sense, cons_senses, variable_names, constraint_names = get_model_matrices(model_open_tepes)
    # solution = np.array(solution)
    # solution_n = solution
    # lb = np.array(lb)
    # ub = np.array(ub)
    model_open_tepes_norm, scaling = normalize_variables(model_open_tepes)
    model_open_tepes_norm.setParam('OutputFlag', 0)
    model_open_tepes_norm.optimize()
    solution_norm = [var.X for var in model_open_tepes_norm.getVars()]
    solution_norm = np.array(solution_norm)
    print(min(solution_norm))   
    print(max(solution_norm))
    print(solution_norm[0])
    scaling_values = list(scaling.values())

    # # if len(scaling_values) == 2:
    # #     scale_factor = scaling_values[1]  # Pick the correct value
    # #     offset = scaling_values[0]  # Offset value
    # #     solution_norm = solution_norm * scale_factor + offset
    # # Compared the solution with the normalized solution
    # print(np.allclose(solution_norm, solution_n, atol=1e-6))
    # A, b, c, co, lb, ub, of_sense, cons_senses, variable_name, constraint_names = get_model_matrices(model_open_tepes_norm)
    # ub = np.array(ub)
    # lb = np.array(lb)
    # print('Model normalized')
    # # upper bound 
    # print(len(np.where(ub == 1)[0]))
    #print(len(np.where(lb == 0)[0]))
    # check_dual_solution(model_open_tepes)
    # importance_constrs, constrs_names = constraints_importance(model_open_tepes)
    # imp_group_constrs = importances_by_groups(importance_constrs, constrs_names, model_open_tepes, 'mean','constraints')
    # m_constraints, m_names = dual_values(model_open_tepes)
    # imp_constrs = importances_by_groups(m_constraints, m_names, model_open_tepes, 'mean','constraints')  
    # save_path = os.path.join(os.getcwd(),'description_data')
    # save_path_constraints = os.path.join(save_path, 'constraints')
    # plot_importances_values(model_open_tepes, m_constraints, m_names, save_path_constraints, 'constraints')
    # importance_vars, vars_names = variables_importance(model_open_tepes)
    # imp_group_vars = importances_by_groups(importance_vars, vars_names, model_open_tepes, 'mean','variables')
    # plot_importances_values(model_open_tepes, importance_vars, vars_names, save_path_constraints, 'variables')
    # zero_importance_constrs, zero_importance_vars, total_vars, total_constrs = zero_importance_values(model_open_tepes)
    # plot_zero_importance(zero_importance_constrs, zero_importance_vars, total_vars, total_constrs, save_path_zero)
