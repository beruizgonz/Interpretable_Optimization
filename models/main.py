import os
import gurobipy as gp
import logging
from datetime import datetime
from Interpretable_Optimization.models.utils_models.utils_modeling import create_original_model, get_model_matrices, \
    save_json, build_model_from_json, compare_models, normalize_features, matrix_sparsification, \
    sparsification_sensitivity_analysis, \
    visual_sparsification_sensitivity, build_dual_model_from_json, visual_join_sparsification_sensitivity, \
    constraint_distance_reduction_sensitivity_analysis, pre_processing_model, constraint_reduction, \
    print_model_in_mathematical_format

logging.basicConfig()
log = logging.getLogger(__name__)
log.setLevel('INFO')

if __name__ == "__main__":

    # ====================================== Defining initial configuration ============================================
    config = {"create_model": {"val": False,
                               "n_variables": 5,
                               "n_constraints": 4},
              "load_model": {"val": True,
                             "load_path": 'GAMS_library',
                             "name": 'transport.mps'},
              "print_mathematical_format": True,
              "verbose": 0,
              "print_detail_sol": True,
              "save_matrices": True,
              "save_original_model": {"val": True,
                                      "save_name": 'testing_transp.mps',
                                      "save_path": 'models_library'},
              "test_normalization": True,
              "test_sparsification": {"val": True,
                                      "threshold": 0.05},
              "test_constraint_red": {"val": True,
                                      "threshold": 0.8},
              "create_presolved": False,
              "sparsification_sensitive_analysis": {"val": True,
                                                    "primal_path": 'primal_sensitivity_analysis',
                                                    "dual_path": 'dual_sensitivity_analysis',
                                                    "max_threshold": 0.3,
                                                    "init_threshold": 0.01,
                                                    "step_threshold": 0.001},
              "euclidian_reduction_sensitive_analysis": {"val": True,
                                                         "primal_path": 'primal_sensitivity_analysis',
                                                         "dual_path": 'dual_sensitivity_analysis',
                                                         "max_threshold": 1.4,
                                                         "init_threshold": 0.1,
                                                         "step_threshold": 0.01},
              "create_dual": True
              }

    # ================================================== Directories to work ===========================================
    # Get the directory of the current script (main.py)
    current_script_dir = os.path.dirname(os.path.abspath(__file__))

    # Navigate up one levels to get to the root of the project
    project_root = os.path.dirname(current_script_dir)

    # paths to work
    data_path = os.path.join(project_root, 'data')
    current_matrices_path = os.path.join(data_path, 'current_matrices')

    # ================================= Creating or loading the original_primal model ==================================
    if config['create_model']['val'] and config['load_model']['val']:
        raise ValueError("Configuration error: 'create_model' and 'load_model' cannot both be True.")

    if config['create_model']['val']:
        log.info(
            f"{str(datetime.now())}: Creating the original_primal with {config['create_model']['n_variables']} variables "
            f"and {config['create_model']['n_constraints']} constraints...")
        original_primal_bp = create_original_model(config['create_model']['n_variables'],
                                                   config['create_model']['n_constraints'])
    if config['load_model']['val']:
        log.info(
            f"{str(datetime.now())}: Loading the original_model...")
        # Specify the path to the MPS file you want to load
        model_to_load = os.path.join(data_path, config['load_model']['load_path'], config['load_model']['name'])

        # Load the model from the MPS file
        original_primal_bp = gp.read(model_to_load)
    # =============================================== Pre-processing the model =========================================
    log.info(
        f"{str(datetime.now())}: Pre-processing the model...")
    original_primal = pre_processing_model(original_primal_bp)

    # ======================================= saving the original_primal model in the path =============================
    if config["save_original_model"]["val"]:
        log.info(
            f"{str(datetime.now())}: Saving the original_primal...")

        s_path = os.path.join(data_path, config["save_original_model"]["save_path"])
        model_to_save = os.path.join(s_path, config["save_original_model"]["save_name"])
        original_primal.write(model_to_save)

    # ================================= Getting matrices and data of the original model ================================
    log.info(
        f"{str(datetime.now())}: Accessing matrix A, right-hand side b, cost function c, and the bounds of "
        f"the original model...")
    A, b, c, lb, ub, of_sense, cons_senses = get_model_matrices(original_primal)

    # ====================================== Saving matrices as json in the path =======================================
    if config['save_matrices']:
        log.info(
            f"{str(datetime.now())}: Saving A, b, c, lb and ub...")
        save_json(A, b, c, lb, ub, of_sense, cons_senses, current_matrices_path)

    # ============================= Creating the primal and the dual models from json files ============================
    log.info(
        f"{str(datetime.now())}: Creating primal model by loading A, b, c, lb and ub...")
    created_primal = build_model_from_json(current_matrices_path)

    log.info(
        f"{str(datetime.now())}: Creating dual model by loading A, b, c, lb and ub...")
    created_dual = build_dual_model_from_json(current_matrices_path)
    A_dual, b_dual, c_dual, lb_dual, ub_dual, of_sense_dual, cons_senses_dual = get_model_matrices(created_dual)

    # ====================================== Printing model in mathematical format =====================================
    if config['print_mathematical_format']:
        print("==================== Original Model before pre-processing ==================== ")
        print_model_in_mathematical_format(original_primal_bp)
        print("==================== Original Model after pre-processing ==================== ")
        print_model_in_mathematical_format(original_primal)
        print("==================== Model created from matrices ==================== ")
        print_model_in_mathematical_format(created_primal)
        print("==================== Dual model created from matrices ==================== ")
        print_model_in_mathematical_format(created_dual)

    # ============ solving all models: original_primal_bp, original_primal, created_primal and created_dual ============
    log.info(
        f"{str(datetime.now())}: Solving the original_primal model (before pre-processing)...")
    original_primal_bp.setParam('OutputFlag', config['verbose'])
    original_primal_bp.optimize()
    original_primal_bp_sol = original_primal_bp.objVal

    log.info(
        f"{str(datetime.now())}: Solving the created_primal model...")
    created_primal.setParam('OutputFlag', config['verbose'])
    created_primal.optimize()
    created_primal_sol = created_primal.objVal

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
    created_dual_sol = created_dual.objVal

    # =============================== Comparing solutions: original_primal X created_primal ============================
    obj_var, dec_var = compare_models(original_primal, created_primal)
    print(f"Create model X original model: The absolute deviation of objective function value is {obj_var} "
          f"and the average deviation of decision variable values is {dec_var}.")

    # printing detailed information about the models
    if config['print_detail_sol']:
        if original_primal_bp.status == 2:
            print("============ Original Model (before pre-processing) ============")
            print("Optimal Objective Value =", original_primal_bp.objVal)
            print("Basic Decision variables: ")
            for var in original_primal_bp.getVars():
                if var.x != 0:
                    print(f"{var.VarName} =", var.x)

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

    # =============================================== Test Normalization ===============================================
    if config['test_normalization']:
        A, b, c, lb, ub, of_sense, cons_senses = get_model_matrices(original_primal)
        A_norm, A_scaler = normalize_features(A)
        b_norm = b / A_scaler
        save_json(A_norm, b_norm, c, lb, ub, of_sense, cons_senses, current_matrices_path)
        created_primal_norm = build_model_from_json(current_matrices_path)
        created_primal_norm.setParam('OutputFlag', config['verbose'])
        created_primal_norm.optimize()
        if created_primal_norm.status == 2:
            print("============ Created Normalized Model ============")
            print("Optimal Objective Value =", created_primal_norm.objVal)
            print("Basic Decision variables: ")
            for var in created_primal_norm.getVars():
                if var.x != 0:
                    print(f"{var.VarName} =", var.x)

    # ================================================ Test Sparsification =============================================
    if config['test_sparsification']['val']:
        A, b, c, lb, ub, of_sense, cons_senses = get_model_matrices(original_primal)
        A_norm, A_scaler = normalize_features(A)
        A_red = matrix_sparsification(config['test_sparsification']['threshold'], A_norm, A)
        save_json(A_red, b, c, lb, ub, of_sense, cons_senses, current_matrices_path)
        created_primal_red = build_model_from_json(current_matrices_path)
        created_primal_red.setParam('OutputFlag', config['verbose'])
        created_primal_red.optimize()

        if created_primal_red.status == 2:
            print("============ Reduced model - Sparsification ============")
            print(f"Threshold: {config['test_sparsification']['threshold']}")
            print("Optimal Objective Value =", created_primal_red.objVal)
            print("Basic Decision variables: ")
            for var in created_primal_red.getVars():
                if var.x != 0:
                    print(f"{var.VarName} =", var.x)

    # ============================================ Test constraint reduction ===========================================
    if config['test_constraint_red']['val']:
        red_model = constraint_reduction(original_primal, config['test_constraint_red']['threshold'],
                                         current_matrices_path)
        red_model.setParam('OutputFlag', config['verbose'])
        red_model.optimize()
        if red_model.status == 2:
            print("============ Reduced model - Constraints reduction ============")
            print(f"Threshold: {config['test_constraint_red']['threshold']}")
            print("Optimal Objective Value =", red_model.objVal)
            print("Basic Decision variables: ")
            for var in red_model.getVars():
                if var.x != 0:
                    print(f"{var.VarName} =", var.x)

    # ===================== Sensitivity analysis on matrix sparsification for different thresholds =====================
    if config['sparsification_sensitive_analysis']['val']:
        eps_p, of_p, dv_p, ind_p = sparsification_sensitivity_analysis(current_matrices_path, original_primal,
                                                                       config['sparsification_sensitive_analysis'],
                                                                       model_to_use='primal')

        eps_d, of_d, dv_d, ind_d = sparsification_sensitivity_analysis(current_matrices_path, created_dual,
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

    # TODO: Automatizar as funcoes de criar dual e primal a partir de matrizes
