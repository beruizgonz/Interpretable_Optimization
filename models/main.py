import os
import gurobipy as gp
import logging
from datetime import datetime
from Interpretable_Optimization.models.utils_models.utils_modeling import create_original_model, get_model_matrices, \
    save_json, build_model_from_json
logging.basicConfig()
log = logging.getLogger(__name__)
log.setLevel('INFO')

if __name__ == "__main__":

    # parameters to run
    config = {"create_model": {"val": True,
                               "n_variables": 10,
                               "n_constraints": 25},
              "load_model": {"val": False,
                             "name": 'original_model.mps'},
              "print_model_sol": True,
              "save_original_model": True,
              "save_matrices": True,
              "create_presolved": False
              }

    # Get the directory of the current script (main.py)
    current_script_dir = os.path.dirname(os.path.abspath(__file__))

    # Navigate up one levels to get to the root of the project
    project_root = os.path.dirname(current_script_dir)

    # path and name to save the original model and matrices
    data_path = os.path.join(project_root, 'data')

    # creating the original_model
    log.info(
        f"{str(datetime.now())}: Creating the original_model with {config['create_model']['n_variables']} variables"
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
    created_model.optimize()
    final_sol = created_model.objVal

    # printing the original model solution
    if config['print_model_sol']:
        print("============ Original Model ============")
        print("Optimal Objective Value =", original_model.objVal)
        for var in original_model.getVars():
            print(f"{var.VarName} =", var.x)

        print("============ Created Model ============")
        print("Optimal Objective Value =", original_model.objVal)
        for var in original_model.getVars():
            print(f"{var.VarName} =", var.x)

    # creating the presolved model
    if config['create_presolved']:
        presolved_model = original_model.presolve()














