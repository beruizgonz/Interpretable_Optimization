import os
import gurobipy as gp
import numpy as np
import matplotlib.pyplot as plt

from gurobipy import GRB, Model

from utils_models.utils_functions import *
from utils_models.standard_model import standard_form1, construct_dual_model, standard_form


# PATHS TO THE DATA
project_root = os.path.dirname(os.getcwd())
model_path = os.path.join(project_root, 'data/GAMS_library', 'AJAX.mps')
GAMS_path = os.path.join(project_root, 'data/GAMS_library')
real_data_path = os.path.join(project_root, 'data/real_data')

GAMS_path_modified = os.path.join(project_root, 'data/GAMS_library_modified')
model_path_modified = os.path.join(GAMS_path_modified, 'AJAX.mps')
real_model_path = os.path.join(real_data_path,  'openTEPES_EAPP_2030_sc01_st1.mps')

# results data
results_folder = os.path.join(project_root, 'figures')
marginal_values = os.path.join(results_folder, 'marginal_values')
save_path_variable = os.path.join(marginal_values, 'variables')
save_path_constraints = os.path.join(marginal_values, 'constraints')
save_simplyfied_constraints = os.path.join(marginal_values, 'simplified_constraints')
save_simplyfied_variables = os.path.join(marginal_values, 'simplified_variables')
save_pareto_variables = os.path.join(marginal_values, 'pareto_variables_importance')
save_pareto_constraints = os.path.join(marginal_values, 'pareto_constraints')


# Create a new Gurobi model
# model = Model("OptimizationProblem")
def new_model():
    model = Model("OptimizationProblem")
    
    # Add variables
    B = model.addVar(vtype=GRB.CONTINUOUS, name="B", lb=0)
    C = model.addVar(vtype=GRB.CONTINUOUS, name="C", lb=0)

    model.update()
    # Set the objective function
    model.setObjective(25 * B +20*C, GRB.MAXIMIZE)

    model.update()

    # Add constraints
    model.addConstr(20 * B + 12 * C <= 1800 , name="Constraint1")
    model.addConstr((1/15) * B + (1/15) * C <= 8 , name="Constraint2")

    model.update()
    return model

def new_model1(): 
    # Create a new model
    model = Model("LP_Optimization")

    # Define variables
    x1 = model.addVar(vtype=GRB.CONTINUOUS, lb=0, ub=1, name="x1")
    x2 = model.addVar(vtype=GRB.CONTINUOUS, lb=0, ub=1, name="x2")
    x3 = model.addVar(vtype=GRB.CONTINUOUS, lb=0, ub=1, name="x3")

    # Set objective function
    model.setObjective(1 * x1 + 5 * x2+ 6*x3 , GRB.MAXIMIZE)

    # Add constraints
    model.addConstr(6 * x1 + 5 * x2 + 8 * x3 <= 16, "Constraint1")
    model.addConstr(10 * x1 + 20 * x2 + 10 * x3 <= 35, "Constraint2")

    # Optimize model
    model.optimize()

    # Display the results
    if model.status == GRB.OPTIMAL:
        print("Optimal solution:")
        print(f"x1 = {x1.x}")
        print(f"x2 = {x2.x}")
        print(f"x3 = {x3.x}")
        print(f"Objective value = {model.objVal}")
    else:
        print("No optimal solution found.")
    return model


def marginal_values_constraints_histogram(model, name, save_path):
    model.setParam('OutputFlag', 0)
    model.optimize()
    
    # Get the shadow prices and corresponding constraints
    shadow_prices = list(model.getAttr('Pi', model.getConstrs()))
    constraints = [constr.ConstrName for constr in model.getConstrs()]
    
    # Filter out zero shadow prices
    filtered_data = [(price, constr) for price, constr in zip(shadow_prices, constraints) if price != 0]
    if not filtered_data:  # If all shadow prices are zero, exit function
        print(f"No non-zero shadow prices for constraints in {name}.")
        return
    
    # Separate shadow prices and constraints after filtering
    shadow_prices, constraints = zip(*filtered_data)
    
    # Normalize the shadow prices
    max_shadow_price = max(shadow_prices)
    shadow_prices = [abs(price / max_shadow_price) for price in shadow_prices]
    
    # Sort in descending order
    sorted_data = sorted(zip(shadow_prices, constraints), reverse=True, key=lambda x: x[0])
    shadow_prices, constraints = zip(*sorted_data)
    
    # Plot the shadow prices
    plt.figure(figsize=(10, 5))
    plt.bar(range(len(shadow_prices)), shadow_prices, color='skyblue', edgecolor='black')
    plt.xticks(range(len(constraints)), constraints, rotation=90)
    plt.xlabel('Constraints')
    plt.ylabel('Normalized marginal values')
    plt.title(f'Marginal values for Constraints: {name}')
    plt.tight_layout()
    save_in = os.path.join(save_path, f'{name}.png')
    plt.savefig(save_in)
    plt.close()



def marginal_values_variables_histogram(model, name, save_path):
    model.setParam('OutputFlag', 0)
    model.optimize()
    
    # Get the reduced costs and corresponding variables
    reduced_costs = list(model.getAttr('RC', model.getVars()))
    constraints = [constr.ConstrName for constr in model.getConstrs()]
    
    # Filter out zero reduced costs
    filtered_data = [(cost, var) for cost, var in zip(reduced_costs, variables) if cost != 0]
    if not filtered_data:  # If all reduced costs are zero, exit function
        print(f"No non-zero reduced costs for variables in {name}.")
        return
    
    # Separate reduced costs and variables after filtering
    reduced_costs, variables = zip(*filtered_data)
    
    # Normalize the reduced costs
    max_reduced_cost = max(reduced_costs)
    reduced_costs = [abs(cost / max_reduced_cost) for cost in reduced_costs]
    
    # Sort in descending order
    sorted_data = sorted(zip(reduced_costs, variables), reverse=True, key=lambda x: x[0])
    reduced_costs, variables = zip(*sorted_data)
    
    # Plot the reduced costs
    plt.figure(figsize=(10, 5))
    plt.bar(range(len(reduced_costs)), reduced_costs, color='skyblue', edgecolor='black')
    plt.xticks(range(len(variables)), variables, rotation=90)
    plt.xlabel('Variables')
    plt.ylabel('Normalized marginal values')
    plt.title(f'Marginal values for Variables: {name}')
    plt.tight_layout()
    save_in = os.path.join(save_path, f'{name}.png')
    plt.savefig(save_in)
    plt.close()

def get_marginal_values(model):
    model.setParam('OutputFlag', 0)
    model.optimize()
    marginal_variables = list(model.getAttr('RC', model.getVars()))
    marginal_constraints = list(model.getAttr('Pi', model.getConstrs()))
    return model.ObjVal, marginal_variables, marginal_constraints

def get_values_constraints(model):
    model.setParam('OutputFlag', 0)
    model.optimize()
    values = [abs(constr.Pi) for constr in model.getConstrs()]
    constraints = [constr.ConstrName for constr in model.getConstrs()]
    return values, constraints

def get_values_variables(model):
    model.setParam('OutputFlag', 0)
    model.optimize()
    values = [var.X for var in model.getVars()]
    variables = [var.VarName for var in model.getVars()]
    return values, variables

def pareto_analysis(save_pareto,name, values, values_name, title_x_axis):
    # Sort data in descending order of values
    sorted_data = sorted(zip(values, values_name), reverse=True, key=lambda x: x[0])
    values, values_name = zip(*sorted_data)

    # Calculate cumulative sum and percentages
    cumsum_values = np.cumsum(values)
    total_sum = sum(values)
    percentage_values = cumsum_values / total_sum * 100

    # Find the index where cumulative percentage exceeds 80%
    threshold_index = np.argmax(percentage_values >= 80)

    # Plot the Pareto chart
    fig, ax1 = plt.subplots(figsize=(10, 6))

    # Bar chart for values
    ax1.bar(values_name, values, color='skyblue', edgecolor='black', label="Values")
    ax1.set_xlabel(f'{title_x_axis}')
    ax1.set_ylabel('Values')
    ax1.set_title(f'Pareto Analysis{name} ({title_x_axis})')
    ax1.tick_params(axis='x', rotation=45)

    # Line chart for cumulative percentage
    ax2 = ax1.twinx()
    ax2.plot(values_name, percentage_values, color='red', marker='o', label="Cumulative %")
    ax2.set_ylabel('Cumulative Percentage')
    ax2.set_ylim(0, 110)
    ax2.axhline(80, color='green', linestyle='--', label="80% Threshold")
    ax2.axvline(x=threshold_index, color='green', linestyle='--')

    # Add a legend
    fig.legend(loc="upper left", bbox_to_anchor=(0.1, 0.9), bbox_transform=ax1.transAxes)

    plt.tight_layout()
    plt.savefig(os.path.join(save_pareto, f'pareto_{name}.png'))
    plt.close()


def marginal_values_constraints1(model, m_constraints, alpha=1e-4): 
    # Count the number of constraints
    num_constraints = len(model.getConstrs())
    constraints = [constr.ConstrName for constr in model.getConstrs()]
    print('Num constraints before removal:', num_constraints)
    
    # Separate positive and negative shadow prices
    positive_shadow_prices = [price for price in m_constraints if price > 0.0]
    negative_shadow_prices = [price for price in m_constraints if price < 0.0]
    
    # Compute maximum absolute shadow prices for positive and negative shadow prices
    max_positive_shadow_price = max(positive_shadow_prices) if positive_shadow_prices else 0.0
    max_negative_shadow_price = min(negative_shadow_prices) if negative_shadow_prices else 0.0  # min since negative
    # max_abs_positive = abs(max_positive_shadow_price)
    # max_abs_negative = abs(max_negative_shadow_price)
    
    print('Max positive shadow price:', max_positive_shadow_price)
    print('Max negative shadow price:', max_negative_shadow_price)
    
    # Set thresholds
    threshold_positive = alpha * max_positive_shadow_price if max_positive_shadow_price > 0 else 0.0
    threshold_negative = alpha * max_negative_shadow_price if max_negative_shadow_price > 0 else 0.0
    
    print('Threshold positive:', threshold_positive)
    print('Threshold negative:', threshold_negative)
    
    # Remove constraints based on thresholds
    for price, constr in zip(m_constraints, constraints):
        abs_price = abs(price)
        if price > 0.0:
            if abs_price < threshold_positive:
                model.remove(model.getConstrByName(constr))
        elif price < 0.0:
            if abs_price < threshold_negative:
                model.remove(model.getConstrByName(constr))
        else:
            # Shadow price is zero; consider removing the constraint
            model.remove(model.getConstrByName(constr))
    model.update()
    
    # Print the number of constraints after removal
    print('Num constraints after removal:', len(model.getConstrs()))
    
    # Optimize the updated model
    model.setParam('OutputFlag', 0)
    model.optimize()
    
    # Check if an optimal solution was found
    if model.status == GRB.OPTIMAL:
        print("Optimal solution:")
        print(f"Objective value = {model.objVal}")
        objective_value = model.objVal
    else:
        print("No optimal solution found.")
        objective_value = np.nan
    
    return len(model.getConstrs()), objective_value

def marginal_values_constraints(model, m_constraints, percentile):
    """
    Simplifies the Gurobi model by removing constraints with reduced costs near zero based on a specified percentile.
    
    Parameters:
    - model: Gurobi model object.
    - m_constraints: List or array of reduced costs (shadow prices) corresponding to each constraint in the model.
    - percentile: The percentile threshold to determine "near zero" reduced costs. 
                  Constraints with absolute reduced costs below this percentile are removed.
    
    Returns:
    - num_constraints_after: Number of constraints remaining after removal.
    - objective_value: Objective value of the simplified model after optimization.
    - simplified_model: The modified Gurobi model.
    
    Raises:
    - ValueError: If the length of m_constraints does not match the number of constraints in the model.
    """
    # Step 1: Create a copy of the model to avoid modifying the original
    simplified_model = model.copy()
    
    # Step 2: Retrieve all constraints from the original model
    constraints = [constr.ConstrName for constr in simplified_model.getConstrs()]
    num_constraints_before = len(constraints)
    print('Num constraints before removal:', num_constraints_before)
    
    # Step 3: Validate the alignment of m_constraints with model constraints
    if len(m_constraints) != num_constraints_before:
        raise ValueError("Length of m_constraints does not match the number of constraints in the model.")
    
    # Step 4: Convert m_constraints to a NumPy array for efficient processing
    m_constraints = np.array(m_constraints, dtype=np.float32)
    
    # Step 5: Compute the absolute reduced costs
    abs_reduced_costs = np.abs(m_constraints)
    
    # Step 6: Determine the threshold value based on the specified percentile
    threshold = np.percentile(abs_reduced_costs, percentile)
    print(f'{percentile}th percentile of absolute reduced costs:', threshold)
    
    # Step 7: Identify constraints with absolute reduced costs below the threshold
    for cost, var in zip(m_constraints, constraints):
        if abs(cost) <= threshold:
            simplified_model.remove(simplified_model.getConstrByName(var))
    simplified_model.update()

    
    # Step 9: Suppress Gurobi output for cleaner logs
    simplified_model.setParam('OutputFlag', 0)
    
    # Step 10: Optimize the simplified model
    simplified_model.optimize()
    
    # Step 11: Retrieve the objective value if the solution is optimal
    if simplified_model.status == GRB.OPTIMAL:
        objective_value = simplified_model.ObjVal
        print('Objective value after removal:', objective_value)
    else:
        print("Model optimization was not successful after constraint removal. Status:", simplified_model.status)
        objective_value = np.nan
    
    # Step 12: Get the number of constraints after removal
    num_constraints_after = len(simplified_model.getConstrs())
    print('Num constraints after removal:', num_constraints_after)
    
    return num_constraints_after, objective_value, simplified_model


def marginal_values_variables(model, m_variables, percentile):
    # Count the number of variables
    simplified_model = model.copy()
    num_variables = len(model.getVars())
    variables = [var.VarName for var in model.getVars()]
    print('Num variables before removal:', num_variables)
    
    # Separate positive and negative reduced costs
    reduced_costs = [cost for cost in m_variables]  
    positive_reduced_costs = [cost for cost in m_variables if cost > 0.0]

    # Convert to numpy array for percentile calculation
    value = np.percentile(reduced_costs, percentile)
    print(f'{percentile}th percentile of positive reduced costs:', value)
    # Delete the variables with reduced costs greater than the 90th percentile
    for cost, var in zip(m_variables, variables):
        if cost > value:
            simplified_model.remove(simplified_model.getVarByName(var))
    simplified_model.update()
    print('Num variables after removal:', len(simplified_model.getVars()))
    # Compute maximum absolute reduced costs for positive and negative reduced costs
    # max_positive_reduced_cost = max(positive_reduced_costs) if positive_reduced_costs else 0.0
    simplified_model.setParam('OutputFlag', 0)
    simplified_model.optimize()
    if simplified_model.status == GRB.OPTIMAL:
        # Get the objective value of the simplified model
        obj = simplified_model.ObjVal
        print('Objective value of the simplified model:', obj)

        # Create a dictionary of variable names and their corresponding values
        simplified_solution = {var.VarName: var.X for var in simplified_model.getVars()}
    else:
        obj = np.nan
        print("Simplified model is not optimal. Objective value set to NaN.")

    # Update the original model with values from the simplified solution
    original_objective = model.getObjective()
    return len(simplified_model.getVars()), obj, simplified_model

def main_marginal_values_constraints(model, min_threshold, max_threshold, step):
    obj, m_variables, m_constraints = get_marginal_values(model)
    thresholds = np.arange(min_threshold, max_threshold, step)
    # inverse the thresholds
    thresholds = thresholds[::-1]
    num_constraints = []
    objective_values = []
    for threshold in thresholds:
        num_constr, obj_value, _ = marginal_values_constraints(model, m_constraints, threshold)
        num_constraints.append(num_constr)
        objective_values.append(obj_value)
    return thresholds, num_constraints, objective_values

def main_marginal_values_variables(model, min_threshold, max_threshold, step):
    obj, m_variables, m_constraints = get_marginal_values(model)
    thresholds = np.arange(min_threshold,max_threshold, step)
    # inverse the thresholds
    thresholds = thresholds[::-1]
    num_variables = []
    objective_values = []
    for threshold in thresholds:
        num_vars, obj_value, _= marginal_values_variables(model, m_variables, threshold)
        num_variables.append(num_vars)
        objective_values.append(obj_value)
    return thresholds, num_variables, objective_values


def test_simplified_solution(model, simplified_model):
    # Create a dictionary of variable names and their corresponding values
    simplified_solution = {var.VarName: var.X for var in simplified_model.getVars()}
    # Update the original model with values from the simplified solution
    original_objective = model.getObjective()

    # Compute the objective value using the simplified solution
    obj_value = 0.0  # Initialize the objective value
    for var in model.getVars():
        if var.VarName in simplified_solution:
            # Use the value from the simplified solution
            value = simplified_solution[var.VarName]
            print(f'Using value for variable {var.VarName}: {value}')
        else:
            # Assign a default value of 0 for missing variables
            value = 0.0
            print(f'Variable {var.VarName} not in simplified solution. Using default value: {value}')
        
        # Add the contribution of this variable to the objective function
        obj_value += value * var.Obj
    print('Objective value using simplified solution:', obj_value)
    return obj_value

def importance_variables(model, variables, m_constraints):
    """
    Get the importance of the variables in the model fuction of the constraints
    """
    A, b, c, co, lb, ub, of_sense, cons_senses, variable_names = get_model_matrices(model)
    # Normalize the problem
    A_norm, b_norm, scalers = normalize_features(A, b)
    # sum coefficients of the matrix A * constraints + objective)* value variableç
    print(A_norm.shape)
    # Convert the constraints to a numpy array
    m_constraints = np.array(m_constraints)
    m_constraints = m_constraints.reshape(-1, 1)
    importance = np.abs(A_norm.T.dot(m_constraints).flatten()) + np.abs(c)
    importance = importance * variables
    return importance


def plot_results(vector_epsilon, vector1, vector2, model_name, folder, plot_type="constraints"):
    """
    Generalized function to plot results for constraints or variables.
    
    Args:
        vector_epsilon (list/np.array): Epsilon values (thresholds).
        vector1 (list/np.array): Objective values.
        vector2 (list/np.array): Number of constraints/variables.
        model_name (str): Name of the model.
        folder (str): Directory to save the plot.
        plot_type (str): Either "constraints" or "variables" for labeling.
    """
    x_axis = np.array(vector_epsilon)
    vector1 = np.array(vector1)
    vector2 = np.array(vector2)

    # Ensure all vectors are of the same length as x_axis by padding with NaN if necessary
    vectors = [vector1, vector2]
    max_length = len(x_axis)
    for i, vec in enumerate(vectors):
        if len(vec) < max_length:
            vectors[i] = np.append(vec, [np.nan] * (max_length - len(vec)))
    vector1, vector2 = vectors

    ### Plot ###
    fig1, axs1 = plt.subplots(2, 1, figsize=(8, 12))
    # reverse the x-axis
    if plot_type == 'variables':
        axs1[0].invert_xaxis()
        init = max(x_axis)
        end = min(x_axis)
    else:
        init = min(x_axis)
        end = max(x_axis)
    # First subplot: Objective Value
    axs1[0].plot(x_axis, vector1, marker='o')
    axs1[0].set_xlim(init,end)  # Adjust as needed
    axs1[0].set_title(f'Objective Value vs. Epsilon ({plot_type.capitalize()})', fontsize=12)
    axs1[0].set_xlabel('Epsilon', fontsize=12)
    axs1[0].set_ylabel('Objective Value', fontsize=12)
    axs1[0].grid(True)

    # Second subplot: Number of Constraints/Variables
    axs1[1].plot(x_axis, vector2, color='orange')
    axs1[1].set_xlim(init, end) # Adjust as needed
    axs1[1].set_title(f'Number of {plot_type.capitalize()} vs. Percentile', fontsize=12)
    axs1[1].set_xlabel('Percentile', fontsize=12)
    axs1[1].set_ylabel(f'Number of {plot_type.capitalize()}', fontsize=12)
    axs1[1].grid(True)

    # Set a title for the entire figure
    fig1.suptitle(f'Results for {model_name} Problem ({plot_type.capitalize()})', fontsize=16)

    # Save the plot
    save_path1 = os.path.join(folder, f'results_{plot_type}_{model_name}.png')
    plt.savefig(save_path1)
    plt.close()

    
if __name__ == '__main__':
    percentile_min_v = 70
    percentile_max_v= 102
    step_v = 2
  
    percentile_min_c = 0
    percentile_max_c= 75
    step_c = 5
    for model_name in os.listdir(GAMS_path_modified):

        model_path = os.path.join(GAMS_path_modified, model_name)
        model = gp.read(model_path)
        name = model_path.split('/')[-1].split('.')[0]

        # marginal_values_constraints_histogram(model, name, save_path_variable)
        # marginal_values_variables_histogram(model, name, save_path_constraints)
        #model = gp.read(model_name)
        # thres, num, obj = main_marginal_values_variables(model, percentile_min_v,percentile_max_v, step_v)

        # plot_results(thres, obj, num, name, save_simplyfied_variables, 'variables')
        # thres, num, obj = main_marginal_values_constraints(model, percentile_min_c, percentile_max_c, step_c)
        # plot_results(thres, obj, num, name, save_simplyfied_constraints, 'constraints')
        #c_values, c_constraints = get_values_constraints(model)
        values_v, names_v = get_values_variables(model)
        values_c, names_c = get_values_constraints(model)
        importance_v = importance_variables(model, values_v, values_c)
        pareto_analysis(save_pareto_variables, f'{name}_importance', importance_v, names_v, 'Variables')
        #pareto_analysis(save_pareto_constraints, name, c_values, c_constraints, 'Constraints')
        #v_values, v_variables = get_values_variables(model)
        #pareto_analysis(save_pareto_variables, name, v_values, v_variables, 'Variables')
    # # model = gp.read(model_path_modified)
    # # #pareto_analysis(model)
    # # thres, num, ob = main_marginal_values_variables(model, percentile_max, percentile_min, step)
    # # plot_results(thres, ob, num, 'AJAX', save_path_constraints, 'variables')
    model = gp.read(model_path_modified)
    values_v, names_v = get_values_variables(model)
    values_c, names_c = get_values_constraints(model)
    importance_v = importance_variables(model, values_v, values_c)
    pareto_analysis(save_pareto_variables, 'AJAX_importance', importance_v, names_v, 'Variables')

    # thres_v, num_v, obj_v = main_marginal_values_variables(model, percentile_min_v, percentile_max_v, step_v)
    # thres_c, num_c, obj_c = main_marginal_values_constraints(model, percentile_min_c, percentile_max_c, step_c)
    # plot_results(thres_v, obj_v, num_v, 'openTEPES_EAPP_2030_sc01_st1', save_simplyfied_variables, 'variables')
    # plot_results(thres_c, obj_c, num_c, 'openTEPES_EAPP_2030_sc01_st1', save_simplyfied_constraints, 'constraints')