import os
import gurobipy as gp
import numpy as np
import matplotlib.pyplot as plt

from gurobipy import GRB, Model

from utils_models.utils_functions import *
from utils_models.standard_model import standard_form1, construct_dual_model, standard_form
from utils_models.utils_plots import pareto_analysis, plot_results


# PATHS TO THE LOAD THE DATA
project_root = os.path.dirname(os.getcwd())
model_path = os.path.join(project_root, 'data/GAMS_library', 'AJAX.mps')
GAMS_path = os.path.join(project_root, 'data/GAMS_library')
real_data_path = os.path.join(project_root, 'data/real_data')

GAMS_path_modified = os.path.join(project_root, 'data/GAMS_library_modified')
model_path_modified = os.path.join(GAMS_path_modified, 'DINAM.mps')
real_model_path = os.path.join(real_data_path,  'openTEPES_EAPP_2030_sc01_st1.mps')

# PATH TO SAVE THE RESULTS
results_folder = os.path.join(project_root, 'figures_new')
marginal_values = os.path.join(results_folder, 'marginal_values')
save_path_variable = os.path.join(marginal_values, 'variables')
save_path_constraints = os.path.join(marginal_values, 'constraints')
save_simplyfied_constraints = os.path.join(marginal_values, 'simplified_constraints')
save_simplyfied_variables = os.path.join(marginal_values, 'simplified_variables')
save_pareto_variables_importance = os.path.join(marginal_values, 'pareto_variables_importance')
save_pareto_constraints = os.path.join(marginal_values, 'pareto_constraints')
save_pareto_variables = os.path.join(marginal_values, 'pareto_variables')

# PARAMETERS FOR THE ANALYSIS
PERCENTILE_MIN_V= 70
PERCENTILE_MAX_V= 102
STEP_V= 2

PERCENTILE_MIN_C = 0
PERCENTILE_MAX_C= 75
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
    return model.ObjVal, marginal_variables, marginal_constraints

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


def marginal_values_handler(model, values, names, percentile, entity_type='constraints'):
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
    if entity_type not in ['constraints', 'variables']:
        raise ValueError("entity_type must be either 'constraints' or 'variables'.")

    num_entities_before = len(names)
    print(f'Num {entity_type} before removal:', num_entities_before)

    # Convert values to a NumPy array for efficient processing
    values = np.array(values, dtype=np.float32)

    # Compute the absolute values
    abs_values = np.abs(values)

    # Determine the threshold value based on the specified percentile
    threshold = np.percentile(abs_values, percentile)
    print(f'{percentile}th percentile of absolute values:', threshold)

    # Identify entities with absolute values below or above the threshold
    for value, name in zip(values, names):
        if (entity_type == 'constraints' and abs(value) <= threshold) or \
           (entity_type == 'variables' and value > threshold):
            if entity_type == 'constraints':
                simplified_model.remove(simplified_model.getConstrByName(name))
            else:
                simplified_model.remove(simplified_model.getVarByName(name))

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

    return num_entities_after, objective_value, simplified_model

def main_marginal_values_handler(model, min_threshold, max_threshold, step, entity_type='constraints'):
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
    obj, m_variables, m_constraints = get_marginal_values(model)
    values, names = (m_constraints, [constr.ConstrName for constr in model.getConstrs()]) if entity_type == 'constraints' else \
                    (m_variables, [var.VarName for var in model.getVars()])

    thresholds = np.arange(min_threshold, max_threshold, step)[::-1]
    num_entities = []
    objective_values = []

    for threshold in thresholds:
        num, obj_value, _ = marginal_values_handler(model, values, names, threshold, entity_type)
        num_entities.append(num)
        objective_values.append(obj_value)

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

    importance = np.abs(coeff + A_norm.T.dot(m_constraints).flatten())
    importance = importance
    return importance

    
if __name__ == '__main__':
    for model_name in os.listdir(GAMS_path_modified):

        model_path = os.path.join(GAMS_path_modified, model_name)
        model = gp.read(model_path)
        name = model_path.split('/')[-1].split('.')[0]

        # marginal_values_constraints_histogram(model, name, save_path_variable)
        # marginal_values_variables_histogram(model, name, save_path_constraints)
        #model = gp.read(model_name)
        thres, num, obj = main_marginal_values_handler(model, PERCENTILE_MIN_C, PERCENTILE_MAX_V, STEP_V, 'variables')
        plot_results(thres, obj, num, name, save_simplyfied_variables, 'variables')
        thres, num, obj = main_marginal_values_handler(model, PERCENTILE_MIN_C, PERCENTILE_MAX_C, STEP_C, 'constraints')
        plot_results(thres, obj, num, name, save_simplyfied_constraints, 'constraints')
        #c_values, c_constraints = get_values_constraints(model)
        # values_v, names_v = get_values_variables(model)
        # values_c, names_c = get_values_constraints(model)
        # importance_v = importance_variables(model, values_v, values_c)
        # pareto_analysis(save_pareto_variables_importance, f'{name}_importance', importance_v, names_v, 'Variables')
        # pareto_analysis(save_pareto_constraints, name, values_c, names_c, 'Constraints')
        # v_values, v_variables = get_values_variables(model)
        # pareto_analysis(save_pareto_variables, name, v_values, v_variables, 'Variables')
    # # model = gp.read(model_path_modified)
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