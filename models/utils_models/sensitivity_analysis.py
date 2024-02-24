# if config['presolve_operations']['eliminate_zero_rows']:
#     log.info(
#         f"{str(datetime.now())}: Presolve operations - eliminate zero rows")
#     print_model_in_mathematical_format(current_model)
#     current_model, feedback_zero_rows = eliminate_zero_rows(current_model, current_matrices_path)
#
# if config['presolve_operations']['eliminate_zero_columns']:
#     log.info(
#         f"{str(datetime.now())}: Presolve operations - eliminate zero columns")
#     current_model.update()
#     print_model_in_mathematical_format(current_model)
#     current_model, feedback_zero_columns = eliminate_zero_columns(current_model, current_matrices_path)
#
# if config['presolve_operations']['eliminate_singleton_equalities']:
#     log.info(
#         f"{str(datetime.now())}: Presolve operations - eliminate singleton equalities")
#     current_model.update()
#     print_model_in_mathematical_format(current_model)
#     current_model, solution_singleton_equalities = eliminate_singleton_equalities(current_model,
#                                                                                   current_matrices_path)
#
# if config['presolve_operations']['eliminate_doubleton_equalities']:
#     log.info(
#         f"{str(datetime.now())}: Presolve operations - eliminate doubleton equalities")
#     current_model.update()
#     print_model_in_mathematical_format(current_model)
#     current_model = eliminate_doubleton_equalities(current_model, current_matrices_path)
#
# if config['presolve_operations']['eliminate_kton_equalities']:
#     log.info(
#         f"{str(datetime.now())}: Presolve operations - eliminate kton equalities")
#     current_model.update()
#     print_model_in_mathematical_format(current_model)
#     current_model, kton_dict = eliminate_kton_equalities(current_model, current_matrices_path, 3)
#
# if config['presolve_operations']['eliminate_singleton_inequalities']:
#     log.info(
#         f"{str(datetime.now())}: Presolve operations - eliminate singleton inequalities")
#     current_model.update()
#     print_model_in_mathematical_format(current_model)
#     current_model, feedback_constraint_single_in, feedback_variable_single_in = eliminate_singleton_inequalities(
#         current_model, current_matrices_path)
#
# if config['presolve_operations']['eliminate_dual_singleton_inequalities']:
#     log.info(
#         f"{str(datetime.now())}: Presolve operations - eliminate dual singleton inequalities")
#     current_model.update()
#     print_model_in_mathematical_format(current_model)
#     current_model, feedback_constraint_single_in, feedback_variable_single_in = (
#         eliminate_dual_singleton_inequalities(current_model, current_matrices_path))
#
# if config['presolve_operations']['eliminate_redundant_columns']:
#     log.info(
#         f"{str(datetime.now())}: Presolve operations - eliminate redundant columns")
#     current_model.update()
#     print_model_in_mathematical_format(current_model)
#     current_model, feedback_constraint_red_col, feedback_variable_red_col = (
#         eliminate_redundant_columns(current_model, current_matrices_path))
#
# if config['presolve_operations']['eliminate_redundant_rows']:
#     log.info(
#         f"{str(datetime.now())}: Presolve operations - eliminate redundant rows")
#     current_model.update()
#     print_model_in_mathematical_format(current_model)
#     current_model, feedback_constraint = eliminate_redundant_rows(current_model, current_matrices_path)
#
# if config['presolve_operations']['eliminate_implied_bounds']:
#     log.info(
#         f"{str(datetime.now())}: Presolve operations - eliminate implied bounds")
#     current_model.update()
#     print_model_in_mathematical_format(current_model)
#     current_model, feedback_constraint = eliminate_implied_bounds(current_model, current_matrices_path)
#
# if config['presolve_operations']['small_coefficient_reduction']:
#     log.info(
#         f"{str(datetime.now())}: Presolve operations - small coefficient reduction")
#     current_model.update()
#     print_model_in_mathematical_format(current_model)
#     current_model, changes = small_coefficient_reduction(current_model)
#
# # ============================================ rhs sensitivity analysis ========================================
# if config['rhs_sensitivity']:
#     log.info(
#         f"{str(datetime.now())}: Right-hand-side sensitivity analysis:")
#     rhs_dec, rhs_inc = rhs_sensitivity(original_primal_bp)
#
# # ======================================== cost vector sensitivity analysis ====================================
# if config['cost_sensitivity']:
#     log.info(
#         f"{str(datetime.now())}: Cost vector sensitivity analysis:")
#     cv_dec, cv_inc = cost_function_sensitivity(original_primal_bp)
#
# # ================================================ Test Sparsification =========================================
# if config['test_sparsification']['val']:
#     log.info(
#         f"{str(datetime.now())}: Results of the sparsification test:")
#     sparsification_test(original_primal, config, current_matrices_path)
#
# # ============================================ Test constraint reduction =======================================
# if config['test_constraint_red']['val']:
#     log.info(
#         f"{str(datetime.now())}: Results of the constraint reduction test:")
#     constraint_reduction_test(original_primal, config, current_matrices_path)
#
#     # =================== Sensitivity analysis on matrix sparsification for different thresholds ===================
#         if config['sparsification_sa']['val']:
#             log.info(
#                 f"{str(datetime.now())}: Sensitivity analysis on matrix sparsification for different thresholds:")
#
#             log.info(f"{datetime.now()}: Processing primal...")
#             eps_p, of_p, dv_p, ind_p, cviol_p, of_dec_p = sparsification_sensitivity_analysis(current_matrices_path,
#                                                                                               original_primal,
#                                                                                               config[
#                                                                                                   'sparsification_sa'],
#                                                                                               model_to_use='primal')
#
#             log.info(f"{datetime.now()}: Processing dual...")
#
#             eps_d, of_d, dv_d, ind_d, cviol_d, of_dec_d = sparsification_sensitivity_analysis(current_matrices_path,
#                                                                                               created_dual,
#                                                                                               config[
#                                                                                                   'sparsification_sa'],
#                                                                                               model_to_use='dual')
#
#             # Assign results to the 'primal' and 'dual' keys using dictionary unpacking
#             sparsification_results[model_name]['primal'] = {
#                 'epsilon': eps_p,
#                 'objective_function': of_p,
#                 'decision_variables': dv_p,
#                 'changed_indices': ind_p,
#                 'constraint_violation': cviol_p,
#                 'of_original_decision': of_dec_p
#             }
#
#             sparsification_results[model_name]['dual'] = {
#                 'epsilon': eps_d,
#                 'objective_function': of_d,
#                 'decision_variables': dv_d,
#                 'changed_indices': ind_d,
#                 'constraint_violation': cviol_d,
#                 'of_original_decision': of_dec_d
#             }
#
#         if config['sparsification_sa']['plots']:
#             title_1 = 'Sparsification'
#             primal_sense = original_primal.ModelSense
#
#             visual_join_sensitivity(eps_p, of_p, dv_p, cviol_p, of_d, dv_d, cviol_d, title_1, primal_sense)
#
#         # ================== Sensitivity analysis on constraint's reduction for different thresholds ===================
#         if config['euclidian_reduction_sensitive_analysis']['val']:
#             eps_p2, of_p2, dv_p2, ind_p2, cviol_p2, of_dec_p2 = (
#                 constraint_distance_reduction_sensitivity_analysis(current_matrices_path, original_primal,
#                                                                    config[
#                                                                        'euclidian_reduction_sensitive_analysis'],
#                                                                    model_to_use='primal'))
#
#             eps_d2, of_d2, dv_d2, ind_d2, cviol_d2, of_dec_d2 = (
#                 constraint_distance_reduction_sensitivity_analysis(current_matrices_path, created_dual,
#                                                                    config[
#                                                                        'euclidian_reduction_sensitive_analysis'],
#                                                                    model_to_use='dual'))
#
#             title_2 = 'Constraints Reduction'
#
#             visual_join_sensitivity(eps_p2, of_p2, dv_p2, cviol_p2, of_d2, dv_d2, cviol_d2, title_2, primal_sense)
#
#         # ======================================== Creating the presolved model ========================================
#         if config['create_presolve']:
#             presolved_model = original_primal.presolve()
#
#         # ============================================== Clearing models  ==============================================
#         # Delete model instances
#         del original_primal_bp, original_primal, created_primal, created_dual
#
#     # Saving the dictionary
#     dict2json(sparsification_results, 'sparsification_results.json')
#
# "rhs_sensitivity": False,
#               "cost_sensitivity": False,
#               "presolve_operations": {"eliminate_zero_rows": False,
#                                       "eliminate_zero_columns": False,
#                                       "eliminate_singleton_equalities": False,
#                                       "eliminate_doubleton_equalities": False,
#                                       "eliminate_kton_equalities": False,
#                                       "eliminate_singleton_inequalities": False,
#                                       "eliminate_dual_singleton_inequalities": False,
#                                       "eliminate_redundant_columns": False,
#                                       "eliminate_redundant_rows": False,
#                                       "eliminate_implied_bounds": True},
#               "test_sparsification": {"val": False,
#                                       "threshold": 0.13},
#               "test_constraint_red": {"val": False,
#                                       "threshold": 0.8},
#               "sparsification_sa": {"val": False,
#                                     "plots": False,
#                                     "prints": False,
#                                     "max_threshold": 0.3,
#                                     "init_threshold": 0.01,
#                                     "step_threshold": 0.001},
#               "euclidian_reduction_sensitive_analysis": {"val": False,
#                                                          "max_threshold": 1.4,
#                                                          "init_threshold": 0.1,
#                                                          "step_threshold": 0.01},
#               "create_presolve": False