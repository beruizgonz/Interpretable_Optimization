import os
import gurobipy as gp
import logging
from datetime import datetime
from Interpretable_Optimization.models.utils_models.utils_modeling import create_original_model, get_model_matrices, \
    save_json, build_model_from_json, compare_models, normalize_features, reduction_features, sensitivity_analysis, \
    visual_sensitivity_analysis

logging.basicConfig()
log = logging.getLogger(__name__)
log.setLevel('INFO')

if __name__ == "__main__":

    # parameters to run
    config = {"create_model": {"val": False,
                               "n_variables": 50,
                               "n_constraints": 25},
              "load_model": {"val": True,
                             "name": 'original_model.mps'},
              "verbose": 0,
              "print_detail_sol": True,
              "save_original_model": False,
              "save_matrices": True,
              "normalize_A": True,
              "Reduction_A": {"val": True,
                              "threshold": 0.1},
              "create_presolved": False,
              "sensitivity_analysis": {"val": True,
                                       "max_threshold": 0.3,
                                       "init_threshold": 0.01,
                                       "step_threshold": 0.001}
              }

    # Get the directory of the current script (main.py)
    current_script_dir = os.path.dirname(os.path.abspath(__file__))

    # Navigate up one levels to get to the root of the project
    project_root = os.path.dirname(current_script_dir)

    # path and name to save the original model and matrices
    data_path = os.path.join(project_root, 'data')

    # creating the original_model
    log.info(
        f"{str(datetime.now())}: Creating the original_model with {config['create_model']['n_variables']} variables "
        f"and {config['create_model']['n_constraints']} constraints...")

    if config['create_model']['val']:
        original_model = create_original_model(config['create_model']['n_variables'],
                                               config['create_model']['n_constraints'])
    else:
        log.info(
            f"{str(datetime.now())}: Loading the original_model...")
        # Specify the path to the MPS file you want to load
        model_to_load = os.path.join(data_path, config['load_model']['name'])

        # Load the model from the MPS file
        original_model = gp.read(model_to_load)

    # solving the original model
    log.info(
        f"{str(datetime.now())}: Solving the original_model...")
    original_model.setParam('OutputFlag', config['verbose'])
    original_model.optimize()
    initial_sol = original_model.objVal

    # saving the original model
    if config["save_original_model"]:
        log.info(
            f"{str(datetime.now())}: Saving the original_model...")
        model_to_save = os.path.join(data_path, "original_model.mps")
        original_model.write(model_to_save)

    # Access constraint matrix A and right-hand side (RHS) vector b of the original model
    log.info(
        f"{str(datetime.now())}: Accessing matrix A, right-hand side b, cost function c, and the bounds of "
        f"the original model...")
    A, b, c, lb, ub = get_model_matrices(original_model)

    # saving the matrix
    if config['save_matrices']:
        log.info(
            f"{str(datetime.now())}: Saving A, b, c, lb and ub...")
        save_json(A, b, c, lb, ub, data_path)

    # creating models from json files
    log.info(
        f"{str(datetime.now())}: Creating model from  A, b, c, lb and ub...")
    created_model = build_model_from_json(data_path)

    # solving created model
    log.info(
        f"{str(datetime.now())}: Solving the created_model...")
    created_model.setParam('OutputFlag', config['verbose'])
    created_model.optimize()
    final_sol = created_model.objVal

    obj_var, dec_var = compare_models(original_model, created_model)
    print(f"The absolute deviation of objective function value is {obj_var} "
          f"and the average deviation of decision variable values is {dec_var}.")

    # printing detailed information about the models
    if config['print_detail_sol']:
        print("============ Original Model ============")
        print("Optimal Objective Value =", original_model.objVal)
        print("Basic Decision variables: ")
        for var in original_model.getVars():
            if var.x != 0:
                print(f"{var.VarName} =", var.x)

        print("============ Created Model ============")
        print("Optimal Objective Value =", created_model.objVal)
        print("Basic Decision variables: ")
        for var in created_model.getVars():
            if var.x != 0:
                print(f"{var.VarName} =", var.x)

    # Normalization
    if config['normalize_A']:
        A_norm, A_scaler = normalize_features(A)
    else:
        A_norm = A.copy()

    # Reduction
    if config['Reduction_A']['val']:
        A_red = reduction_features(config['Reduction_A']['threshold'], A_norm, A)

    # creating the presolved model
    if config['create_presolved']:
        presolved_model = original_model.presolve()

    # Sensitivity analysis
    if config['sensitivity_analysis']['val']:
        eps, of, dv = sensitivity_analysis(data_path, original_model, config['sensitivity_analysis'])

    visual_sensitivity_analysis(eps, of, dv)










