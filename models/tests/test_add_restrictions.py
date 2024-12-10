import os
import gurobipy as gp
import sys
import pandas as pd


parent_dir = os.path.dirname(os.getcwd())
sys.path.append(parent_dir)

from models.opts import parse_opts
from models.utils_models.presolvepsilon_class import PresolvepsilonOperations
from models.utils_models.utils_functions import *
from models.utils_models.standard_model import standard_form, construct_dual_model, standard_form2

# Path to the data folder
parent_dir = os.path.dirname(os.getcwd())
GAMS_models_folder = os.path.join(parent_dir, 'data/GAMS_library_modified')
excel_path = os.path.join(parent_dir, 'data/GAMS_library/GAMS_Models_Summary.xlsx')


def add_new_restriction(model_path, excel_objective): 
    """
    Add new restriction to the model
    """
    # Read the model from the file
    model = gp.read(model_path)
    model_name = model_path.split('/')[-1]
    
    # Get the objective function value from the Excel file
    df = pd.read_excel(excel_objective)
    objective = df[df['Name of LP'] == model_name]['Optimal Solution Value'].values[0]
    # Get the objective expression from the model
    obj_expr = model.getObjective()
    
    # Copy the model
    new_model = model.copy()
    
    # Recreate the objective expression using the variables from the new model
    new_obj_expr = gp.LinExpr()
    for i in range(obj_expr.size()):
        var = obj_expr.getVar(i)
        coeff = obj_expr.getCoeff(i)
        new_obj_expr += coeff * new_model.getVarByName(var.VarName)
    
    if model.ModelSense == gp.GRB.MINIMIZE:
        print('Minimize')
        new_model.addConstr(new_obj_expr <= 5* objective)
    elif model.ModelSense == gp.GRB.MAXIMIZE:
        print('Maximize')
        new_model.addConstr(new_obj_expr >= 2* objective)
    # print the new constraints
    # print(new_model.getConstrs())
    new_model.update()
    
    return new_model

def add_new_restrictions_variables(model_path):
    """
    Add new restrictions to the model. The restrictions I want to add are based on the variables of the model when I solve it
    I want to add for each variable a restriction that the variable must be less than 10*variable_value. For variable value I have to solve
    """
    model = gp.read(model_path)
    model_name = model_path.split('/')[-1]
    # Copy the model
    new_model = model.copy()
    # Solve the model
    model.optimize()
    # Get the values of the variables
    variables = new_model.getVars()

    # Add a new constraint for each variable using the values from the original model
    for var in variables:
        original_var = model.getVarByName(var.VarName)
        new_model.addConstr(var <= 10 * original_var.X)

    # Update the model to integrate the new constraints
    new_model.update()

    return new_model


def compare_bounds(model_add, model):
    """
    Compare the bounds of two models
    """
    standard_model = standard_form(model)
    standard_model_add = standard_form(model_add)
    lb, ub = calculate_bounds(standard_model)
    lb_add, ub_add = calculate_bounds(standard_model_add)
    return lb, ub, lb_add, ub_add

if __name__ == '__main__': 
    problem = 'PRODSP.mps'
    model_path = os.path.join(GAMS_models_folder, problem)

    model_objective = add_new_restrictions_variables(model_path)
    model = gp.read(model_path)
    
    lb, ub, lb_add, ub_add = compare_bounds(model_objective, model)
    print(ub)
    print(ub_add)
    #print(np.all(ub == ub_add)) 
    # count = 0
    # for model_name in os.listdir(GAMS_models_folder):
    #     model_path = os.path.join(GAMS_models_folder, model_name)
    #     model_objective = add_new_restriction(model_path, excel_path)
    #     model = gp.read(model_path)
    #     num_constraints = model.getAttr('NumConstrs')   
    #     num_constraints_objective = model_objective.getAttr('NumConstrs')
    #     # Compare number of constraints
    
        
    #     # # Compare objective values
    #     # print(f"Model: {model_name}")
    #     # # optimize the models
    #     model.setParam('OutputFlag', 0)
    #     model_objective.setParam('OutputFlag', 0)
    #     model.optimize()
    #     model_objective.optimize()
    #     if model.ObjVal == model_objective.ObjVal:
    #         count += 1
    #print(f"Number of models with the same objective value: {count}")