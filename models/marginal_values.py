import os
import gurobipy as gp
import numpy as np
import matplotlib.pyplot as plt

from gurobipy import GRB, Model

from utils_models.utils_functions import *
from utils_models.utils_plots import pareto_analysis, plot_results
from utils_models.standard_model import*
#from description_data.description_open_TEPES import *

# PATHS TO THE LOAD THE DATA
project_root = os.path.dirname(os.getcwd())
model_path = os.path.join(project_root, 'data/GAMS_library', 'AJAX.mps')
GAMS_path = os.path.join(project_root, 'data/GAMS_library')
real_data_path = os.path.join(project_root, 'data/real_data')

GAMS_path_modified = os.path.join(project_root, 'data/GAMS_library_modified')
model_path_modified = os.path.join(GAMS_path_modified, 'DINAM.mps')
real_model_path = os.path.join(real_data_path,  'openTEPES_EAPP_2030_sc01_st1.mps')

# PATH TO SAVE THE RESULTS
figures_folder = os.path.join(project_root, 'figures_new')
results_folder = os.path.join(project_root, 'results_new')
marginal_values = os.path.join(figures_folder, 'marginal_values')
save_path_variable = os.path.join(marginal_values, 'variables')
save_path_constraints = os.path.join(marginal_values, 'constraints')
save_simplified_constraints = os.path.join(marginal_values, 'real_problems/simplified_constraints_percentage')
save_simplified_variables = os.path.join(marginal_values, 'real_problems/simplified_variables_percentage')
save_simplified_importance_variables = os.path.join(marginal_values, 'simplified_importance_variables')
save_pareto_variables_importance = os.path.join(marginal_values, 'pareto_variables_importance')
save_pareto_constraints = os.path.join(marginal_values, 'pareto_constraints')
save_pareto_variables = os.path.join(marginal_values, 'pareto_variables')
save_folder_variables = os.path.join(results_folder, 'marginal_values/real_problems/simplified_variables_percentage')
save_folder_constraints = os.path.join(results_folder, 'marginal_values/real_problems/simplified_constraints_epsilon')

# PARAMETERS FOR THE ANALYSIS
PERCENTILE_MIN_V= 0
PERCENTILE_MAX_V= 55
STEP_V= 2

PERCENTILE_MIN_C = 0
PERCENTILE_MAX_C= 55
STEP_C = 5

# DEFINE THE FUNCTIONS 

def get_marginal_values(model):
    """
    Parameters:
    - model: Gurobi model object.
    Returns:
    - objective_value: Objective value of the model.
    - marginal_variables: List of reduced costs (marginal values) corresponding to each variable in the model.
    - marginal_constraints: List of shadow prices (marginal values) corresponding to each constraint in the model.
    """
    model.setParam('OutputFlag', 0)
    model.optimize()
    marginal_variables = list(model.getAttr('RC', model.getVars()))
    marginal_constraints = list(model.getAttr('Pi', model.getConstrs()))
    variables = [var.X for var in model.getVars()]
    return model.ObjVal, variables, marginal_variables, marginal_constraints

def get_importance_values(model):
    """
    Parameters:
    - model: Gurobi model object.
    Returns:
    - objective_value: Objective value of the model.
    - marginal_variables: List of reduced costs (marginal values) corresponding to each variable in the model.
    - marginal_constraints: List of shadow prices (marginal values) corresponding to each constraint in the model.
    """
    model = normalize_variables(model)
    model.setParam('OutputFlag', 0)
    model.optimize()
    marginal_variables = list(model.getAttr('RC', model.getVars()))
    marginal_constraints = list(model.getAttr('Pi', model.getConstrs()))
    solution = [var.X for var in model.getVars()]
    A, b, c, co, lb, ub, of_sense, cons_senses, variable_names, constrs_names = get_model_matrices(model)
    constraints_values = A.dot(solution)
    marginal_constraints = np.array(marginal_constraints)
    importance_constrs = np.abs(constraints_values * marginal_constraints)

    c = np.array(c)
    importance_vars = np.abs(c - A.T.dot(marginal_constraints).flatten())
    return model.ObjVal, importance_vars, importance_constrs

def get_values_constraints(model):
    """
    Parameters:
    - model: Gurobi model object.
    Returns:
    - values: List of shadow prices (marginal values) corresponding to each constraint in the model.
    - constraints: List of constraint names.
    """
    model.setParam('OutputFlag', 0)
    model.optimize()
    values = [abs(constr.Pi) for constr in model.getConstrs()]
    constraints = [constr.ConstrName for constr in model.getConstrs()]
    return values, constraints

def get_values_variables(model):
    """
    Parameters:
    - model: Gurobi model object.
    Returns:
    - values: List of reduced costs (marginal values) corresponding to each variable in the model.
    - variables: List of variable names.
    """
    model.setParam('OutputFlag', 0)
    model.optimize()
    values = [abs(var.RC) for var in model.getVars()]
    variables = [var.VarName for var in model.getVars()]
    return values, variables


def marginal_values_histogram(model, name, save_path, data_type='constraints'):
    """
    Generate a histogram for marginal values of constraints or variables.

    Parameters:
        model: Gurobi model object.
        name: Name for the plot and file.
        save_path: Directory to save the plot.
        data_type: 'constraints' or 'variables' to specify which data to plot.
    """
    model.setParam('OutputFlag', 0)
    model.optimize()

    if data_type == 'constraints':
        values = list(model.getAttr('Pi', model.getConstrs()))
        labels = [constr.ConstrName for constr in model.getConstrs()]
        y_label = 'Normalized marginal values'
        title = f'Marginal values for Constraints: {name}'
    elif data_type == 'variables':
        values = list(model.getAttr('RC', model.getVars()))
        labels = [var.VarName for var in model.getVars()]
        y_label = 'Normalized reduced costs'
        title = f'Marginal values for Variables: {name}'
    else:
        raise ValueError("data_type must be either 'constraints' or 'variables'.")

    # Filter out zero values
    filtered_data = [(val, label) for val, label in zip(values, labels) if val != 0]
    if not filtered_data:
        print(f"No non-zero {data_type} values for {name}.")
        return

    # Separate values and labels after filtering
    values, labels = zip(*filtered_data)

    # Normalize the values
    max_value = max(values)
    normalized_values = [abs(val / max_value) for val in values]

    # Sort in descending order
    sorted_data = sorted(zip(normalized_values, labels), reverse=True, key=lambda x: x[0])
    normalized_values, labels = zip(*sorted_data)

    # Plot the values
    plt.figure(figsize=(10, 5))
    plt.bar(range(len(normalized_values)), normalized_values, color='skyblue', edgecolor='black')
    plt.xticks(range(len(labels)), labels, rotation=90)
    plt.xlabel(data_type.capitalize())
    plt.ylabel(y_label)
    plt.title(title)
    plt.tight_layout()

    # Save the plot
    save_in = os.path.join(save_path, f'{name}.png')
    plt.savefig(save_in)
    plt.close()


def remove_variables_flexibility(model, values, names, percentile, metric = 'Percentage'): 
    """
    Remove variables with marginal values below a specified percentile.
    Also, determine which constraints involve these variables.
    """
    # Extract model matrices
    A, b, c, co, lb, ub, of_sense, cons_senses, variable_names, constrs_names = get_model_matrices(model)

    # Ensure values and names match in length
    if len(values) != len(names):
        raise ValueError(f"Length mismatch: values ({len(values)}) and names ({len(names)})")

    if metric == 'Percentage':
        abs_values = np.abs(values)
        # Sort the values in ascending order
        order = np.argsort(abs_values)
        # The way of simplifying is to delete the % with the smallest values
        variables_to_remove = []
        constrs_to_remove = []
        var_indices_to_remove = []
        for i in range(int(len(order)*percentile/100)):
            variables_to_remove.append(names[order[i]])
            var_indices_to_remove.append(order[i])
    elif metric == 'Epsilon':
        # Identify variables to remove based on threshold
        abs_values = np.abs(values)
        mask = abs_values < percentile
        variables_to_remove = np.array(names)[mask].tolist()

        # Get indices of variables to remove
        var_indices_to_remove = np.where(mask)[0]

        if len(var_indices_to_remove) == 0:
            print("No variables to remove.")
            return [], []  # Early exit if no variables meet the criteria

    # Ensure A has the correct shape
    if A.shape[1] != len(names):
        raise ValueError(f"Matrix A column size ({A.shape[1]}) does not match names length ({len(names)})")
    min_max_indices = {}  # Dictionary to track min/max indices for each row
    # Efficient constraint removal using sparse operations
    constraints_to_remove = {}
    if issparse(A):  
        A = A.tocsc()  # Convert to CSC for fast column slicing
        var_indices_to_remove = np.array(var_indices_to_remove, dtype=int)
        affected_rows = A[:, var_indices_to_remove].nonzero()[0]  
        row_indices, col_indices = A[:, var_indices_to_remove].nonzero()
        
        # Create a dictionary to track min/max for each row
        for row, col in zip(row_indices, col_indices):
            constr_name = constrs_names[row]
            actual_col = var_indices_to_remove[col]  # Map to original indices
            if row not in min_max_indices:
                min_max_indices[constr_name] = [actual_col, actual_col]  # Initialize (min, max)
            else:
                min_max_indices[constr_name][0] = min(min_max_indices[row][0], actual_col)  # Update min
                min_max_indices[constr_name][1] = max(min_max_indices[row][1], actual_col)  # Update max

    else:  # Dense matrix case
        constraints_to_remove = np.where(np.any(A[:, var_indices_to_remove] != 0, axis=1))[0]
    # Remove the variables from the model
  # Remove variables efficiently in bulk
    print(f"Removing {len(variables_to_remove)} variables from the model.")
    model.remove([model.getVarByName(var_name) for var_name in variables_to_remove])
    model.update()

    # Update constraint senses based on min/max index
    for constr_name, (min_idx, max_idx) in min_max_indices.items():
        constr = model.getConstrByName(constr_name)
        abs_min, abs_max = abs(min_idx), abs(max_idx)
        if min_idx < 0 and abs_min > abs_max:
            constr.Sense = 'G'  # Greater than or equal
        elif max_idx > 0 and abs_max > abs_min:
            constr.Sense = 'L'  # Less than or equal
        elif abs_min == abs_max:
            constr.Sense = 'L'  # Less than or equal
        else:
            constr.Sense = 'E'  # Equality

    model.update()
    return model, variables_to_remove

def marginal_values_handler(model, values, names, percentile, entity_type='constraints', type_simplyfied='Percentile'):
    """
    Simplifies the Gurobi model by removing constraints or variables based on a specified percentile.

    Parameters:
    - model: Gurobi model object.
    - values: List or array of marginal values (e.g., reduced costs or shadow prices).
    - names: List of names corresponding to the constraints or variables.
    - percentile: The percentile threshold to determine "near zero" marginal values.
    - entity_type: Either 'constraints' or 'variables'.

    Returns:
    - num_entities_after: Number of constraints or variables remaining after removal.
    - objective_value: Objective value of the simplified model after optimization.
    - simplified_model: The modified Gurobi model.
    """
    simplified_model = model.copy()

    # Validate entity_type
    if entity_type not in ['constraints', 'variables', 'importance_variables']:
        raise ValueError("entity_type must be either 'constraints' or 'variables'.")

    num_entities_before = len(names)
    print(f'Num {entity_type} before removal:', num_entities_before)

    # Convert values to a NumPy array for efficient processing
    values = np.array(values, dtype=np.float32)

    # Compute the absolute values
    abs_values = np.abs(values)
    get_remove_name = []
    if type_simplyfied == 'Percentile':
    # Determine the threshold value based on the specified percentile
        threshold = np.percentile(abs_values, percentile)
        print(f'{percentile}th percentile of absolute values:', threshold)

        # Identify entities with absolute values below or above the threshold
        for value, name in zip(values, names):
            # Check if the current entity is a constraint and meets the threshold condition
            if entity_type == 'constraints' and abs(value) <= threshold:
                # Remove the constraint by name from the simplified model
                simplified_model.remove(simplified_model.getConstrByName(name))
            # Check if the current entity is a variable and meets the threshold condition
            elif entity_type == 'variables' and value > threshold:
                # Remove the variable by name from the simplified model
                simplified_model.remove(simplified_model.getVarByName(name))
            elif entity_type == 'importance_variables' and value <= threshold:
                simplified_model.remove(simplified_model.getVarByName(name))
    elif type_simplyfied == 'Percentage':
        # order the values from the smallest to the largest
        order = np.argsort(abs_values)
        # The way of simplifying is to delete the % with the smallest values
       
        if entity_type == 'constraints':
                for i in range(int(len(order)*percentile/100)):
                    simplified_model.remove(simplified_model.getConstrByName(names[order[i]]))
                    get_remove_name.append(names[order[i]])
        elif entity_type == 'variables':
            simplified_model = model.copy()
            simplified_model, variable_remove = remove_variables_flexibility(simplified_model, values, names, percentile, 'Percentage')
            get_remove_name = variable_remove
        elif entity_type == 'importance_variables':
            simplified_model.remove(simplified_model.getVarByName(names[order[i]]))
    elif type_simplyfied == 'Epsilon':
        order = np.argsort(abs_values)
        # Delay the values that are less than epsilon
        if entity_type == 'constraints':
            for i in range(len(order)):
                if abs_values[order[i]] < percentile:
                    simplified_model.remove(simplified_model.getConstrByName(names[order[i]]))
                    get_remove_name.append(names[order[i]])
        elif entity_type == 'variables':
            variables_to_remove, constrs_remove = remove_variables_flexibility(simplified_model, values, names, percentile)
            # for i in range(len(variables_to_remove)):
            #     simplified_model.remove(simplified_model.getVarByName(variables_to_remove[i]))
            #     get_remove_name.append(variables_to_remove[i])
            # for i in range(len(constrs_remove)):
            #     simplified_model.remove(simplified_model.getConstrByName(constrs_remove[i]))
        elif entity_type == 'importance_variables':
            for i in range(len(order)):
                if abs_values[order[i]] < percentile:
                    simplified_model.remove(simplified_model.getVarByName(names[order[i]]))

    simplified_model.update()

    simplified_model.setParam('OutputFlag', 0)
    simplified_model.optimize()
    if simplified_model.status == GRB.OPTIMAL:
        objective_value = simplified_model.ObjVal
        print('Objective value after removal:', objective_value)
    else:
        print(f"Model optimization was not successful after {entity_type} removal. Status:", simplified_model.status)
        objective_value = np.nan

    num_entities_after = len(simplified_model.getConstrs()) if entity_type == 'constraints' else len(simplified_model.getVars())
    print(f'Num {entity_type} after removal:', num_entities_after)
    print(f'Num constraints after removal:', len(simplified_model.getConstrs()))
    # delay simplyfied model
    del simplified_model
    return num_entities_after, objective_value, get_remove_name

def main_marginal_values_handler(model, model_name, save_folder, min_threshold, max_threshold, step, metric = 'importance', save_result = True, entity_type='constraints', type_simplyfied='Percentile'):
    """
    Iterates over thresholds to analyze the effect of removing constraints or variables based on marginal values.

    Parameters:
    - model: Gurobi model object.
    - min_threshold: Minimum percentile threshold.
    - max_threshold: Maximum percentile threshold.
    - step: Step size for threshold iteration.
    - entity_type: Either 'constraints' or 'variables'.

    Returns:
    - thresholds: Array of thresholds.
    - num_entities: List of remaining constraints or variables at each threshold.
    - objective_values: List of objective values at each threshold.
    """
    if metric =='marginal':
        obj,_, m_variables, m_constraints = get_marginal_values(model)
    elif metric == 'importance':
        _, m_variables, m_constraints = get_importance_values(model)
    if entity_type == 'constraints':
        # For constraints, return shadow prices and constrain)t names
        values = m_constraints
        names = [constr.ConstrName for constr in model.getConstrs()]
    elif entity_type == 'variables':
        # For variables, return reduced costs and variable names
        values = m_variables
        names = [var.VarName for var in model.getVars()]
    elif entity_type == 'importance_variables': 
        values = importance_variables(model, m_variables, m_constraints)
        names = [var.VarName for var in model.getVars()]
    if type_simplyfied == 'Epsilon':
        thresholds = []
        threshold = min_threshold
        while threshold < max_threshold:
            thresholds.append(threshold)
            threshold += threshold* step
            thresholds.append(threshold)
        thresholds = np.array(thresholds)
    else: 
        thresholds = np.arange(min_threshold, max_threshold, step)
    num_entities = []
    objective_values = []
    names_remove = []
    for threshold in thresholds:
        print(f'Processing threshold: {threshold}')
        num, obj_value, name = marginal_values_handler(model, values, names, threshold, entity_type, type_simplyfied)
        num_entities.append(num)
        objective_values.append(obj_value)
        names_remove.append(name)

    results = {'thresholds': thresholds, 'num_entities': num_entities, 'objective_values': objective_values, 'names_remove': names_remove}  
    if save_result: 
        # Save the results in a json file   
        save_path = os.path.join(save_folder, f'{metric}_{model_name}_{entity_type}.json')
        dict2json(results, save_path)
    return thresholds, num_entities, objective_values


def importance_variables(model, variables, m_constraints):
    """
    Get the importance of the variables in the model fuction of the constraints
    Parameters:
    - model: Gurobi model object.
    - variables: List of values of the variables.
    - m_constraints: List of values of the marginal values of the constraints.
    """
    A, b, c, co, lb, ub, of_sense, cons_senses, variable_names = get_model_matrices(model)
    A_norm, b_norm, scalers = normalize_features(A, b)

    m_constraints = np.array(m_constraints)
    m_constraints = m_constraints.reshape(-1, 1)
    coeff = np.array([var.Obj for var in model.getVars()])
    c = np.array(c)
    importance = np.abs(c - A.T.dot(m_constraints).flatten()) 
    return importance

def test_marginal_variables(model): 
    obj, m_variables, m_constraints = get_marginal_values(model)
    # get the first m_variables
    first_variables = m_variables[-2:-1]
    first_variables_names = [var.VarName for var in model.getVars()][-2:-1]
    print(first_variables)

def compared_importance_marginal_variables(model):
    standard_model = standard_form_e2(model)
    obj, m_variables, m_constraints = get_marginal_values(standard_model)
    importance = importance_variables(standard_model, m_variables, m_constraints)
    vbasis = standard_model.getAttr("VBasis")  # Get the basis status for all variables
    basis_variables = np.where(np.array(vbasis) != 0)[0]  # Obtain the basis variables
    # Obtain the importance of the basis variables
    importance_basis = importance[basis_variables]
    print(importance_basis)
    m_variables = np.array(m_variables)
    m_varialbes_basis = m_variables[basis_variables]
    print(m_varialbes_basis)



if __name__ == '__main__':
    # for model_name in os.listdir(GAMS_path_modified):

    #     model_path = os.path.join(GAMS_path_modified, model_name)
    #     model = gp.read(model_path)
    #     name = model_path.split('/')[-1].split('.')[0]

    #     # marginal_values_constraints_histogram(model, name, save_path_variable)
    #     # marginal_values_variables_histogram(model, name, save_path_constraints)
    #     # thres, num, obj = main_marginal_values_handler(model, PERCENTILE_MIN_V, PERCENTILE_MAX_V, STEP_C, 'variables', 'Percentages')
    #     # plot_results(thres, obj, num, name, save_simplified_variables, 'variables')
    #     # thres, num, obj = main_marginal_values_handler(model, PERCENTILE_MIN_C, PERCENTILE_MAX_C, STEP_C, 'constraints','Percentages')
    #     # plot_results(thres, obj, num, name, save_simplified_constraints, 'constraints')
    #     tresh, num, obj = main_marginal_values_handler(model, PERCENTILE_MIN_V, PERCENTILE_MAX_V, STEP_V, 'importance_variables', 'Percentages')
    #     plot_results(tresh, obj, num, name, save_simplified_importance_variables, 'variables')
    #     # #c_values, c_constraints = get_values_constraints(model)
        # values_v, names_v = get_values_variables(model)
        # values_c, names_c = get_values_constraints(model)
        # importance_v = importance_variables(model, values_v, values_c)
        # pareto_analysis(save_pareto_variables_importance, f'{name}_importance', importance_v, names_v, 'Variables')
        # pareto_analysis(save_pareto_constraints, name, values_c, names_c, 'Constraints')
        # v_values, v_variables = get_values_variables(model)
        # pareto_analysis(save_pareto_variables, name, v_values, v_variables, 'Variables')
    model = gp.read(real_model_path)
    model_name = real_model_path.split('/')[-1].split('.')[0]
    print(model_name)
    model = standard_form_e2(model)
    model.setParam('OutputFlag', 0)
    thres, num, obj = main_marginal_values_handler(model, model_name, save_folder_variables, PERCENTILE_MIN_C, PERCENTILE_MAX_V, STEP_V, 'importance', True, 'variables', 'Percentage')
    #plot_results(thres, obj, num, 'openTEPES_EAPP_2030_sc01_st1', save_simplified_variables, 'variables')
    #print(os.path.exists(save_simplified_constraints))
    #thres, num, obj = main_marginal_values_handler(model, model_name, save_folder_constraints, PERCENTILE_MIN_C, PERCENTILE_MAX_C, STEP_V, 'importance',True, 'constraints', 'Percentage')
    #plot_results(thres, obj, num, 'openTEPES_EAPP_2030_sc01_st1', save_simplified_constraints, 'constraints')
    # obj, variables, m_variables, m_constraints = get_marginal_values(model)
    # importance = importance_variables(model, variables, m_constraints)
    # print(importance)
    # compared_importance_marginal_variables(model)
    # # #pareto_analysis(model)
    # # thres, num, ob = main_marginal_values_variables(model, percentile_max, percentile_min, step)
    # # plot_results(thres, ob, num, 'AJAX', save_path_constraints, 'variables')
    # model = gp.read(model_path_modified)
    # values_v, names_v = get_values_variables(model)
    # values_c, names_c = get_values_constraints(model)
    # importance_v = importance_variables(model, values_v, values_c)
    # pareto_analysis(save_pareto_variables, 'AJAX_importance', importance_v, names_v, 'Variables')

    # thres_v, num_v, obj_v = main_marginal_values_variables(model, percentile_min_v, percentile_max_v, step_v)
    # thres_c, num_c, obj_c = main_marginal_values_constraints(model, percentile_min_c, percentile_max_c, step_c)
    # plot_results(thres_v, obj_v, num_v, 'openTEPES_EAPP_2030_sc01_st1', save_simplyfied_variables, 'variables')
    # plot_results(thres_c, obj_c, num_c, 'openTEPES_EAPP_2030_sc01_st1', save_simplyfied_constraints, 'constraints')