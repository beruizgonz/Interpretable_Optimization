import os
import gurobipy as gp
import logging
from datetime import datetime
from Interpretable_Optimization.models.utils_models.utils_modeling import create_original_model, get_model_matrices, \
    save_json, build_model_from_json, compare_models, normalize_features, reduction_features, sensitivity_analysis, \
    visual_sensitivity_analysis, build_dual_model_from_json

logging.basicConfig()
log = logging.getLogger(__name__)
log.setLevel('INFO')

if __name__ == "__main__":

    # ====================================== Defining initial configuration ============================================
    config = {"create_model": {"val": True,
                               "n_variables": 50,
                               "n_constraints": 30},
              "load_model": {"val": False,
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
                                       "sens_path": 'sensitivity_analisys',
                                       "max_threshold": 0.3,
                                       "init_threshold": 0.01,
                                       "step_threshold": 0.001},
              "create_dual": True
              }

    # ================================================== Directories to work ===========================================
    # Get the directory of the current script (main.py)
    current_script_dir = os.path.dirname(os.path.abspath(__file__))

    # Navigate up one levels to get to the root of the project
    project_root = os.path.dirname(current_script_dir)

    # path and name to save the original model and matrices
    data_path = os.path.join(project_root, 'data')

    # ================================= Creating or loading the original_primal model ==================================
    log.info(
        f"{str(datetime.now())}: Creating the original_primal with {config['create_model']['n_variables']} variables "
        f"and {config['create_model']['n_constraints']} constraints...")

    if config['create_model']['val']:
        original_primal = create_original_model(config['create_model']['n_variables'],
                                                config['create_model']['n_constraints'])
    else:
        log.info(
            f"{str(datetime.now())}: Loading the original_model...")
        # Specify the path to the MPS file you want to load
        model_to_load = os.path.join(data_path, config['load_model']['name'])

        # Load the model from the MPS file
        original_primal = gp.read(model_to_load)

    # ======================================= saving the original_primal model in the path =============================
    if config["save_original_model"]:
        log.info(
            f"{str(datetime.now())}: Saving the original_primal...")
        model_to_save = os.path.join(data_path, "original_primal.mps")
        original_primal.write(model_to_save)

    # ========================== Getting matrices A, b, c and the bound of the original model ==========================
    log.info(
        f"{str(datetime.now())}: Accessing matrix A, right-hand side b, cost function c, and the bounds of "
        f"the original model...")
    A, b, c, lb, ub = get_model_matrices(original_primal)

    # ====================================== Saving matrices as json in the path =======================================
    if config['save_matrices']:
        log.info(
            f"{str(datetime.now())}: Saving A, b, c, lb and ub...")
        save_json(A, b, c, lb, ub, data_path)

    # ============================= Creating the primal and the dual models from json files ============================
    log.info(
        f"{str(datetime.now())}: Creating primal model by loading A, b, c, lb and ub...")
    created_primal = build_model_from_json(data_path)

    log.info(
        f"{str(datetime.now())}: Creating dual model by loading A, b, c, lb and ub...")
    created_dual = build_dual_model_from_json(data_path)
    A_dual, b_dual, c_dual, lb_dual, ub_dual = get_model_matrices(created_dual)

    # ====================== solving all models: original_primal, created_primal and created_dual ======================
    log.info(
        f"{str(datetime.now())}: Solving the original_primal model...")
    original_primal.setParam('OutputFlag', config['verbose'])
    original_primal.optimize()
    original_primal_sol = original_primal.objVal

    log.info(
        f"{str(datetime.now())}: Solving the created_primal model...")
    created_primal.setParam('OutputFlag', config['verbose'])
    created_primal.optimize()
    created_primal_sol = created_primal.objVal

    log.info(
        f"{str(datetime.now())}: Solving the created_dual model...")
    created_dual.setParam('OutputFlag', config['verbose'])
    created_dual.optimize()
    try:
        created_dual_sol = created_dual.objVal
    except:
        # created_dual.computeIIS()
        # created_dual.write("model.ilp")
        # print("Infeasible constraints written to 'model.ilp'")
        print("Status:", created_dual.status)

    # =============================== Comparing solutions: original_primal X created_primal ============================
    obj_var, dec_var = compare_models(original_primal, created_primal)
    print(f"The absolute deviation of objective function value is {obj_var} "
          f"and the average deviation of decision variable values is {dec_var}.")

    # printing detailed information about the models
    if config['print_detail_sol']:
        print("============ Original Model ============")
        print("Optimal Objective Value =", original_primal.objVal)
        print("Basic Decision variables: ")
        for var in original_primal.getVars():
            if var.x != 0:
                print(f"{var.VarName} =", var.x)

        print("============ Created Model ============")
        print("Optimal Objective Value =", created_primal.objVal)
        print("Basic Decision variables: ")
        for var in created_primal.getVars():
            if var.x != 0:
                print(f"{var.VarName} =", var.x)

    # ==================================== Normalizing Matrix A from original_primal ===================================
    if config['normalize_A']:
        A_norm, A_scaler = normalize_features(A)
    else:
        A_norm = A.copy()

    # ============================== Reducing the normalized matrix A from original_primal =============================
    if config['Reduction_A']['val']:
        A_red = reduction_features(config['Reduction_A']['threshold'], A_norm, A)

    # =========================== Sensitivity analysis: Reducing A for different thresholds ============================
    if config['sensitivity_analysis']['val']:
        sens_data = os.path.join(data_path, config['sensitivity_analysis']['sens_path'])
        eps, of, dv, ind = sensitivity_analysis(sens_data, original_primal, config['sensitivity_analysis'])
        visual_sensitivity_analysis(eps, of, dv)

    # ========================================== Creating the presolved model ==========================================
    if config['create_presolved']:
        presolved_model = original_primal.presolve()
