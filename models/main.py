import os
import gurobipy as gp
import logging
from datetime import datetime
from Interpretable_Optimization.models.utils_models.utils_modeling import create_original_model, get_model_matrices, \
    save_json, build_model_from_json, compare_models, normalize_features, matrix_sparsification, \
    sparsification_sensitivity_analysis, \
    visual_sparsification_sensitivity, build_dual_model_from_json, visual_join_sparsification_sensitivity, \
    constraint_distance_reduction_sensitivity_analysis

logging.basicConfig()
log = logging.getLogger(__name__)
log.setLevel('INFO')

if __name__ == "__main__":

    # ====================================== Defining initial configuration ============================================
    config = {"create_model": {"val": True,
                               "n_variables": 25,
                               "n_constraints": 10},
              "load_model": {"val": False,
                             "name": 'original_model.mps'},
              "verbose": 1,
              "print_detail_sol": True,
              "save_matrices": True,
              "save_original_model": {"val": True,
                                      "save_name": 'original_primal.mps'},
              "normalize_A": True,
              "Reduction_A": {"val": True,
                              "threshold": 0.1},
              "create_presolved": False,
              "sparsification_sensitive_analysis": {"val": False,
                                                    "primal_path": 'primal_sensitivity_analysis',
                                                    "dual_path": 'dual_sensitivity_analysis',
                                                    "max_threshold": 0.3,
                                                    "init_threshold": 0.01,
                                                    "step_threshold": 0.001},
              "euclidian_reduction_sensitive_analysis": {"val": True,
                                                         "primal_path": 'primal_sensitivity_analysis',
                                                         "dual_path": 'dual_sensitivity_analysis',
                                                         "max_threshold": 1,
                                                         "init_threshold": 0.1,
                                                         "step_threshold": 0.01},
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
    if config["save_original_model"]["val"]:
        log.info(
            f"{str(datetime.now())}: Saving the original_primal...")
        model_to_save = os.path.join(data_path, config["save_original_model"]["save_name"])
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
        print("Status:", created_dual.status)

    # =============================== Comparing solutions: original_primal X created_primal ============================
    obj_var, dec_var = compare_models(original_primal, created_primal)
    print(f"The absolute deviation of objective function value is {obj_var} "
          f"and the average deviation of decision variable values is {dec_var}.")

    # printing detailed information about the models
    if config['print_detail_sol']:
        if original_primal.status == 2:
            print("============ Original Model ============")
            print("Optimal Objective Value =", original_primal.objVal)
            print("Basic Decision variables: ")
            for var in original_primal.getVars():
                if var.x != 0:
                    print(f"{var.VarName} =", var.x)

        if created_primal.status == 2:
            print("============ Created Model ============")
            print("Optimal Objective Value =", created_primal.objVal)
            print("Basic Decision variables: ")
            for var in created_primal.getVars():
                if var.x != 0:
                    print(f"{var.VarName} =", var.x)

        if created_dual.status == 2:
            print("============ Created Dual ============")
            print("Optimal Objective Value =", created_dual.objVal)
            print("Basic Decision variables: ")
            for var in created_dual.getVars():
                if var.x != 0:
                    print(f"{var.VarName} =", var.x)

    # ==================================== Normalizing Matrix A from original_primal ===================================
    if config['normalize_A']:
        A_norm, A_scaler = normalize_features(A)
    else:
        A_norm = A.copy()

    # ============================== Reducing the normalized matrix A from original_primal =============================
    if config['Reduction_A']['val']:
        A_red = matrix_sparsification(config['Reduction_A']['threshold'], A_norm, A)

    # ===================== Sensitivity analysis on matrix sparsification for different thresholds =====================
    if config['sparsification_sensitive_analysis']['val']:
        primal_data = os.path.join(data_path, config['sparsification_sensitive_analysis']['primal_path'])
        eps_p, of_p, dv_p, ind_p = sparsification_sensitivity_analysis(primal_data, original_primal,
                                                                       config['sparsification_sensitive_analysis'])

        dual_data = os.path.join(data_path, config['sparsification_sensitive_analysis']['dual_path'])
        eps_d, of_d, dv_d, ind_d = sparsification_sensitivity_analysis(dual_data, created_dual,
                                                                       config['sparsification_sensitive_analysis'],
                                                                       model_to_use='dual')
        # visual_sparsification_sensitivity(eps_p, of_p, dv_p)
        visual_join_sparsification_sensitivity(eps_p, of_p, dv_p, of_d, dv_d)

    # ===================== Sensitivity analysis on matrix sparsification for different thresholds =====================
    if config['euclidian_reduction_sensitive_analysis']['val']:
        primal_data = os.path.join(data_path, config['euclidian_reduction_sensitive_analysis']['primal_path'])
        eps_p, of_p, dv_p, ind_p = (
            constraint_distance_reduction_sensitivity_analysis(primal_data, original_primal,
                                                               config[
                                                                   'euclidian_reduction_sensitive_analysis']))

        dual_data = os.path.join(data_path, config['euclidian_reduction_sensitive_analysis']['dual_path'])
        eps_d, of_d, dv_d, ind_d = (
            constraint_distance_reduction_sensitivity_analysis(dual_data, created_dual,
                                                               config[
                                                                   'euclidian_reduction_sensitive_analysis'],
                                                               model_to_use='dual'))
        # visual_sparsification_sensitivity(eps_p, of_p, dv_p)
        visual_join_sparsification_sensitivity(eps_p, of_p, dv_p, of_d, dv_d)

    # ========================================== Creating the presolved model ==========================================
    if config['create_presolved']:
        presolved_model = original_primal.presolve()
