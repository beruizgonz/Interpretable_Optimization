import os 
import gurobipy as gp
import sys

parent_dir = os.path.dirname(os.getcwd())
sys.path.append(parent_dir)

from utils_models.presolvepsilon_class import PresolvepsilonOperations
from utils_models.utils_functions import *
from utils_models.standard_model import standard_form, construct_dual_model, standard_form_e1


def main_bounds(folder):
    # Create a pandas dataframe
    rows = []
    objective_bounds = 0  # Initialize objective_bounds to 0
    objective_standard = 0

    for model_name in os.listdir(folder):
        # Skip Excel files if present in the folder
        if model_name.endswith('.xlsx'):
            continue

        # Construct the full path of the model
        model_path = os.path.join(folder, model_name)
        name = model_name.split('.')[0]  # Extract the base name of the model

        try:
            # Load the model
            model = gp.read(model_path)  # Assuming the model is in a valid format (like MPS or LP)
            model.setParam('OutputFlag', 0)
            model.optimize()
            print(model.objVal)
          
            if model.Status == gp.GRB.OPTIMAL:
                objective = model.objVal
                print(f'Objective value of the original model: {model.objVal}')
            standard_model = standard_form_e1(model)
            standard_model.setParam('OutputFlag', 0)
            standard_model.optimize()
            #print(standard_model.objVal)
           
            # Uncomment these if you want to calculate bounds later
        
            A_sparse, b, c, co, lb_new, ub_new, of_sense, cons_sense, variable_names= calculate_bounds_candidates(standard_model, None)
            bounds_model = build_model(A_sparse, b, c, co, lb_new, ub_new, of_sense, cons_sense, variable_names)
            bounds_model.setParam('OutputFlag', 0)
            bounds_model.optimize()

            dual_model = construct_dual_model(standard_model)
            dual_model.setParam('OutputFlag', 0)
            dual_model.optimize()
            if bounds_model.Status == gp.GRB.OPTIMAL:
                objective_bounds = bounds_model.objVal
                print(f'Objective value with bounds: {objective_bounds}')

            # # Check if the optimization status is optimal
            if standard_model.Status == gp.GRB.OPTIMAL:
                objective_standard = standard_model.objVal
                print(f'Objective value without bounds: {objective_standard}')
            else:
                objective_standard = 0  # Set to 0 if not optimal

            if dual_model.Status == gp.GRB.OPTIMAL:
                objective_dual = dual_model.objVal
                print(f'Objective value of the dual model: {objective_dual}')
          
            #del model, standard_model, bounds_model, dual_model

        except gp.GurobiError as e:
            print(f"Error reading or optimizing model {model_name}: {e}")
            objective_standard = 0  # If there's an error, set the objective to 0

        # Add a new row with model name, standard objective, and bounds objective

        new_row = [name, objective, objective_standard, objective_bounds, objective_dual]
        rows.append(new_row)
        
    # Create a DataFrame with the collected rows
    results = pd.DataFrame(rows, columns=['Model', 'Normal', 'Standard', 'Bounds', 'Dual'])
    #order the results by alphabetical order of the model name
    results = results.sort_values(by='Model')
    # Save the DataFrame to an Excel file
    try:
        results.to_excel('results_bounds.xlsx', index=False)
        print("Results successfully saved to 'results_bounds.xlsx'")
    except Exception as e:
        print(f"Failed to save results to Excel: {e}")



project_root = os.path.dirname(os.getcwd())
project_root = os.path.dirname(project_root)
model_path = os.path.join(project_root, 'data/GAMS_library', 'INDUS.mps')
GAMS_path = os.path.join(project_root, 'data/GAMS_library')
GAMS_path_modified = os.path.join(project_root, 'data/GAMS_library_modified')
model_path_modified = os.path.join(GAMS_path_modified, 'DINAM.mps')

def standar():
    GAMS_path = os.path.join(project_root, 'data/GAMS_library')
    print(len(os.listdir(GAMS_path)))
    for model_name in os.listdir(GAMS_path):
        if model_name.endswith('.xlsx'):
            continue
        model_path = os.path.join(GAMS_path, model_name)
        model = gp.read(model_path)
        # Solve the problem in canonical form and standard form
        original_primal, track_elements = canonical_form(model)
        standard_model = standard_form(model)
        print_model(original_primal)
        print_model(standard_model)
        original_primal.setParam('OutputFlag', 0)
        standard_model.setParam('OutputFlag', 0)
        original_primal.optimize()
        standard_model.optimize()
            # Check if the objective values are the same
        #if original_primal.Status == GRB.OPTIMAL and standard_model.Status == GRB.OPTIMAL:
        print(model_name[-1], original_primal.objVal,  standard_model.objVal)


if __name__ == '__main__':
    main_bounds(GAMS_path_modified)
    # # results = nested_dict()
    # # for model_name in os.listdir(GAMS_path):
    # #     model_path = os.path.join(GAMS_path, model_name)
    # #     if model_name.endswith('.xlsx'):
    # #         continue
    # #     model = gp.read(model_path)
    # #     original_primal, track_elements = canonical_form(model)

    # #     presolve = load_class(original_primal)
    # #     results_problem =  sensitivity_analysis(original_primal, presolve)
    # #     results[model_name] = results_problem

    # # dict2json(results, 'epsilon_results.json')

    """
    ------------------------CALCULATE THE BOUNDS OF THE MODEL------------------------
    """
    # model_name = model_path_modified
    # print(model_name)
    # model = gp.read(model_name)

    # model.setParam('OutputFlag', 0)
    # model.optimize()
    # print(model.objVal)
    # # #print_model_in_mathematical_format(model)
    # # # Get the model in standard form
    # # #A, b, obj = parse_mps(model_name)
    # standard_model = standard_form(model)
    # # # #canonical_model, _ = canonical_form(model)
    # # # # canonical_model.setParam('OutputFlag', 0)
    # # #print_model_in_mathematical_format(standard_model)
    # standard_model.setParam('OutputFlag', 0)
    # standard_model.optimize()
    # print(standard_model.objVal)
    # # for var in standard_model.getVars():
    # #     print(f'{var.VarName}: {var.X}')
    # # # print(standard_model.objVal)
    # # print(standard_model.objVal)
    # # dual_model = construct_dual_model(standard_model)
    # # # # dual_model1 = construct_dual_model1(standard_model)
    # # # # print_model_in_mathematical_format(dual_model1)
    # # # # dual_model1.optimize()
    # # # # print(dual_model1.objVal)
    # # dual_model.setParam('OutputFlag', 0)
    # # dual_model.optimize()
    # # print(dual_model.objVal)


    # #print_model_in_mathematical_format(dual_model)
    # # for var in standard_model.getVars():
    # #     print(f'{var.VarName}: {var.X}')
    # # #print_model_in_mathematical_format(standard_model)
    # lb, ub = calculate_bounds(standard_model)
    # print(f'Lower bounds: {lb}')
    # print(f'Upper bounds: {ub}')

    # Count the number of infinite upper bounds
    #print(f'Number of infinite upper bounds: {len(t[0])}')
    # # # # # # See if lb <= ub
    # # # # # print(all(lb_new <= ub_new))
    # path = os.path.join(os.getcwd(), 'model_matrices.json')
    # model1 = build_model_from_json(path)
    # # # print_model_in_mathematical_format(model)
    # model1.setParam('OutputFlag', 0)
    # model1.optimize()
    # print(model1.objVal)
    # for var in model.getVars():
    #     print(f'{var.VarName}: {var.X}')
    # Compute the sensitivity analysis

    #main_bounds(GAMS_path_modified)
