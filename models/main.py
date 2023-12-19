import os
import gurobipy as gp
import logging
import numpy as np
from datetime import datetime
from Interpretable_Optimization.models.utils_models.utils_functions import create_original_model, get_model_matrices, \
    save_json, build_model_from_json, compare_models, normalize_features, matrix_sparsification, \
    sparsification_sensitivity_analysis, build_dual_model_from_json, \
    constraint_distance_reduction_sensitivity_analysis, pre_processing_model, constraint_reduction, \
    print_model_in_mathematical_format, visual_join_sensitivity, \
    measuring_constraint_infeasibility, quality_check, sparsification_test, constraint_reduction_test

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
                             "name": 'TRNSPORT.mps'},
              "print_mathematical_format": False,
              "verbose": 0,
              "print_detail_sol": False,
              "save_matrices": True,
              "save_original_model": {"val": True,
                                      "save_name": 'testing_transp.mps',
                                      "save_path": 'models_library'},
              "test_normalization": True,
              "test_sparsification": {"val": True,
                                      "threshold": 0.13},
              "test_constraint_red": {"val": True,
                                      "threshold": 0.8},
              "create_presolved": False,
              "sparsification_sa": {"val": True,
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
    log.info(
        f"{str(datetime.now())}: Preparing directories...")

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
            f"{str(datetime.now())}: Loading the original_model - {config['load_model']['name']}...")
        # Specify the path to the MPS file you want to load
        model_to_load = os.path.join(data_path, config['load_model']['load_path'], config['load_model']['name'])

        # Set the OutputFlag to 0 to suppress Gurobi output
        gp.setParam('OutputFlag', 0)

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
        log.info(
            f"{str(datetime.now())}: Printing detailed mathematical formulations...")

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
        f"{str(datetime.now())}: Solving the original_primal model (after pre-processing)...")
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

    # printing detailed information about the models
    if config['print_detail_sol']:

        obj_var, dec_var = compare_models(original_primal, created_primal)
        print(f"Create model X original model: The absolute deviation of objective function value is {obj_var} "
              f"and the average deviation of decision variable values is {dec_var}.")

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

    # ===================================================  Normalization ===============================================
    log.info(
        f"{str(datetime.now())}: Solving the normalized primal model...")

    A, b, c, lb, ub, of_sense, cons_senses = get_model_matrices(original_primal)
    A_norm, A_scaler = normalize_features(A)
    b_norm = b / A_scaler
    save_json(A_norm, b_norm, c, lb, ub, of_sense, cons_senses, current_matrices_path)
    created_primal_norm = build_model_from_json(current_matrices_path)
    created_primal_norm.setParam('OutputFlag', config['verbose'])
    created_primal_norm.optimize()

    # ==================================================== Quality check ===============================================

    quality_check(original_primal_bp, original_primal, created_primal, created_dual, created_primal_norm,
                  tolerance=1e-6)
    log.info(
        f"{str(datetime.now())}: Quality check passed...")

    # ================================================ Test Sparsification =============================================
    if config['test_sparsification']['val']:
        log.info(
            f"{str(datetime.now())}: Results of the sparsification test:")
        sparsification_test(original_primal, config, current_matrices_path)

    # ============================================ Test constraint reduction ===========================================
    if config['test_constraint_red']['val']:
        log.info(
            f"{str(datetime.now())}: Results of the constraint reduction test:")
        constraint_reduction_test(original_primal, config, current_matrices_path)

    # ============================================== Test variable reduction ===========================================

    # TODO

    # ===================== Sensitivity analysis on matrix sparsification for different thresholds =====================
    if config['sparsification_sa']['val']:
        log.info(
            f"{str(datetime.now())}: Sensitivity analysis on matrix sparsification for different thresholds:")

        eps_p, of_p, dv_p, ind_p, cviol_p, of_dec_p = sparsification_sensitivity_analysis(current_matrices_path,
                                                                                          original_primal,
                                                                                          config[
                                                                                              'sparsification_sa'],
                                                                                          model_to_use='primal')

        eps_d, of_d, dv_d, ind_d, cviol_d, of_dec_d = sparsification_sensitivity_analysis(current_matrices_path,
                                                                                          created_dual,
                                                                                          config[
                                                                                              'sparsification_sa'],
                                                                                          model_to_use='dual')
        title = 'Sparsification'
        visual_join_sensitivity(eps_p, of_p, dv_p, cviol_p, of_d, dv_d, cviol_d)

    # ==================== Sensitivity analysis on constraint's reduction for different thresholds =====================
    if config['euclidian_reduction_sensitive_analysis']['val']:
        eps_p2, of_p2, dv_p2, ind_p2, cviol_p2, of_dec_p2 = (
            constraint_distance_reduction_sensitivity_analysis(current_matrices_path, original_primal,
                                                               config[
                                                                   'euclidian_reduction_sensitive_analysis'],
                                                               model_to_use='primal'))

        eps_d2, of_d2, dv_d2, ind_d2, cviol_d2, of_dec_d2 = (
            constraint_distance_reduction_sensitivity_analysis(current_matrices_path, created_dual,
                                                               config[
                                                                   'euclidian_reduction_sensitive_analysis'],
                                                               model_to_use='dual'))
        # visual_sparsification_sensitivity(eps_p, of_p, dv_p)
        visual_join_sensitivity(eps_p2, of_p2, dv_p2, cviol_p2, of_d2, dv_d2, cviol_d2)

    # ========================================== Creating the presolved model ==========================================
    if config['create_presolved']:
        presolved_model = original_primal.presolve()
