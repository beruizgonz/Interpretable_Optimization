import gurobipy as gp
from gurobipy import GRB
import numpy as np

import gurobipy as gp
from gurobipy import GRB
import numpy as np
from scipy.sparse import csr_matrix

from models.utils_models import get_model_matrices, save_json



def construct_dual_model1(standard_model):
    # Extract primal model data
    A, b, c, co, lb, ub, of_sense, cons_senses, variable_names = get_model_matrices(standard_model)

    # Initialize the dual model
    dual_model = gp.Model('DualModel')

    # Define dual variables corresponding to primal constraints
    dual_vars = []
    for i, sense in enumerate(cons_senses):
        constr_name = f'y_{i}'
        if sense == '>':
            # Primal constraint is '>='
            # Dual variable is non-negative
            y = dual_model.addVar(lb=0, name=constr_name)
        elif sense == '<':
            # Primal constraint is '<='
            # Dual variable is non-positive
            y = dual_model.addVar(ub=0, name=constr_name)
        elif sense == '=':
            # Primal constraint is '='
            # Dual variable is unrestricted
            y = dual_model.addVar(lb=-GRB.INFINITY, ub=GRB.INFINITY, name=constr_name)
        else:
            raise ValueError(f"Unknown constraint sense: {sense}")
        dual_vars.append(y)

    dual_model.update()

    # Set the dual objective function
    dual_obj = gp.LinExpr()
    for i in range(len(b)):
        dual_obj += b[i] * dual_vars[i]
    dual_obj += co  # Add constant term from primal objective if any

    # Determine dual objective sense (opposite of primal)
    if of_sense == GRB.MINIMIZE:
        dual_obj_sense = GRB.MAXIMIZE
    else:
        dual_obj_sense = GRB.MINIMIZE

    dual_model.setObjective(dual_obj, dual_obj_sense)

    # Transpose of A
    A_transpose = A.transpose()

    # Add dual constraints corresponding to primal variables
    for j in range(len(c)):
        constr_expr = gp.LinExpr()
        for i in range(len(b)):
            coeff = A_transpose[j, i]
            constr_expr += coeff * dual_vars[i]
        # Handle primal variable bounds
        var_name = variable_names[j]
        if lb[j] == 0 and ub[j] >= GRB.INFINITY:
            # Primal variable x_j >= 0
            # Dual constraint: <= c_j
            dual_model.addConstr(constr_expr <= c[j], name=f'dual_constr_{var_name}')
        elif lb[j] == -GRB.INFINITY and ub[j] == 0:
            # Primal variable x_j <= 0
            # Dual constraint: >= c_j
            dual_model.addConstr(constr_expr >= c[j], name=f'dual_constr_{var_name}')
        elif lb[j] == -GRB.INFINITY and ub[j] >= GRB.INFINITY:
            # Primal variable x_j unrestricted
            # Dual constraint: == c_j
            dual_model.addConstr(constr_expr == c[j], name=f'dual_constr_{var_name}')
        else:
            # Primal variable has finite bounds, need to handle accordingly
            # For simplicity, raise an exception
            raise NotImplementedError("Primal variable bounds other than non-negative or non-positivity are not handled in this example.")
    
    dual_model.update()
    return dual_model

def calculate_bounds(model):
    """
    Repeatedly calculate the bounds of variables in a Gurobi model based on the constraints
    until the lower and upper bounds converge (i.e., do not change between iterations).

    Parameters:
    - model: Gurobi model object, the optimization model from which to calculate the bounds.

    Returns:
    - lb_new: numpy array, the updated lower bounds of the variables.
    - ub_new: numpy array, the updated upper bounds of the variables.
    """
    # Extract model matrices and data
    A, b, c, co, lb, ub, of_sense, cons_sense, variable_names = get_model_matrices(model)
    n_constraints, n_variables = A.shape

    # Convert A to a dense array if it's a CSR matrix
    if scipy.sparse.isspmatrix_csr(A):
        A_dense = A.toarray()
    else:
        A_dense = A  # Assume it's already a dense array

    # Initialize bounds
    lb_new = lb.copy()
    ub_new = ub.copy()

    # Prepare b as a column vector
    b_vector = np.array(b).reshape(-1, 1)  # Shape: (n_constraints, 1)
    # Repeat the b_vector n_variables times to match the shape of A_dense
    #b_vector = np.tile(b_vector, (1, n_variables))

    converged = False
    iteration = 0

    while iteration < 1:
        iteration += 1
        print(f"Iteration {iteration}")
        # Step 1: Compute new upper bounds (ub_new_next)
        CL_matrix_ub = np.tile(lb_new, (n_variables, 1))
        np.fill_diagonal(CL_matrix_ub, 0)
        ub_numerator = b_vector - (A_dense @ CL_matrix_ub.T)
        ub_denominator = A_dense
        ub_matrix = np.where(ub_denominator > 0, ub_numerator / ub_denominator, np.nan)
        # lb_matrix = np.where(ub_denominator < 0, ub_numerator / ub_denominator, np.nan)
        ub_new_next = np.nanmin(ub_matrix, axis=0)
        ub_new_next = np.where(np.isnan(ub_new_next), np.inf, ub_new_next)
        ub_new_next[ub_new_next <= 0] = np.inf  # Optionally set negative upper bounds to zero

        # # Step 2: Compute new lower bounds (lb_new_next)
        CL_matrix_lb = np.tile(ub_new_next, (n_variables, 1))
        np.fill_diagonal(CL_matrix_lb, 0)
        lb_numerator = b_vector - (A_dense @ CL_matrix_lb.T)
        lb_denominator = A_dense
        # Calculate lb_new_next as the maximum over constraints (ignoring NaNs)
        lb_matrix = np.where(lb_denominator > 0, lb_numerator / lb_denominator, 0)
        lb_new_next = np.nanmax(lb_matrix, axis=0)
        print(lb_new_next)

        # if (np.allclose(lb_new, lb_new_next, rtol=1e-6, atol=1e-6) and
        #     np.allclose(ub_new, ub_new_next, rtol=1e-6, atol=1e-6)):
        #     print("Convergence reached.")
        #     converged = True
        # else:
        #     #Update lb_new and ub_new for the next iteration
        #     lb_new = lb_new_next
        #     ub_new = ub_new_next
        # The upper bounds must be greater than or equal to the lower bounds
    #ub_new_next[ub_new_next <= 0] = np.inf  # Optionally set negative upper bounds to zero
    # print(ub_new_next)
        # Optionally save intermediate results to a JSON file (for tracking progress)
    ub_new = ub_new_next
    save_json(A, b, c, lb_new, ub_new, of_sense, cons_sense, "model_matrices.json", co, variable_names)

    # Return the updated bounds once convergence is reached
    return lb_new, ub_new


def calculate_bounds1(model): 
        """
            Performs bound strengthening on decision variables of a linear programming (LP) problem.
            This operation aims to tighten the lower and upper bounds of variables based on the
            constraint matrix, right-hand side values, and existing bounds, potentially reducing the
            feasible region and improving the efficiency of optimization algorithms.

            The process involves analyzing each constraint to determine how the bounds of a variable
            can be tightened without altering the feasible set of the LP problem. This is done by
            calculating the minimum activity for each constraint (excluding the variable under
            consideration) and using it to adjust the variable's bounds.
        """
        bound_strengthened = 0
        A, b, c, co, lb, ub, of_sense, cons_sense, variable_names = get_model_matrices(model)
        n_constraints, n_variables = A.shape

        # Convert A to a dense array if it's a CSR matrix
        if scipy.sparse.isspmatrix_csr(A):
            A_dense = A.toarray()
        else:
            A_dense = A  # Assume it's already a dense array
        num_rows, num_cols = A_dense.shape
        min_activity_matrix = np.where(A_dense < 0, -A_dense* lb,
                                        np.where(A_dense > 0, -A_dense *ub, 0))
        max_activity_matrix = np.where(A_dense> 0, -A_dense * lb,
                                       np.where(A_dense < 0, A_dense * ub, 0))

        # Initialize an empty array for the complementary_min_activity
        complementary_min_activity = np.zeros_like(min_activity_matrix)
        complementary_max_activity = np.zeros_like(max_activity_matrix)

        # Use broadcasting to perform the operation for each column
        for j in range(num_cols):
            # Create a copy of min_activity_matrix and set the jth column to 0
            modified_matrix_min = np.copy(min_activity_matrix)
            modified_matrix_max = np.copy(max_activity_matrix)

            modified_matrix_min[:, j] = 0  # Exclude column j from the sum
            modified_matrix_max[:, j] = 0  # Exclude column j from the sum

            # Sum over rows to calculate the complementary min activity for each element not considering column j
            complementary_min_activity[:, j] = modified_matrix_min.sum(axis=1)
            complementary_max_activity[:, j] = modified_matrix_max.sum(axis=1)

        # Initialize the new_upper_bound_matrix with +inf
        new_upper_bound_matrix = np.full_like(A_dense, np.inf, dtype=np.float64)
        new_lower_bound_matrix = np.full_like(A_dense, 0, dtype=np.float64)

        # Iterate over each element in A to calculate the new upper bounds
        for i in range(num_rows):
            for j in range(num_cols):
                a_ij = A_dense[i, j]
                if a_ij < 0:
                    # Calculate new upper bound based on complementary min activity
                    new_upper_bound_matrix[i, j] = (-b[i] - complementary_min_activity[i, j]) / (-1*a_ij)
                elif a_ij > 0:
                    # Calculate new upper bound based on complementary min activity
                    new_lower_bound_matrix[i, j] = (-1*b[i] - complementary_min_activity[i, j]) / (-1*a_ij)

        # Initialize a vector to hold the final new upper bounds for each variable
        final_new_upper_bounds = np.full(num_cols, -np.inf)
        final_new_lower_bounds = np.full(num_cols, -np.inf)

        for j in range(num_cols):
            final_new_upper_bounds[j] = np.min(new_upper_bound_matrix[:, j])
            final_new_lower_bounds[j] = np.max(new_lower_bound_matrix[:, j])

        # Update the original upper bounds if the new calculated bounds are tighter
        for j in range(num_cols):
            if (final_new_upper_bounds[j] < ub[j]) and (final_new_upper_bounds[j] > lb[j]):
                ub[j] = final_new_upper_bounds[j]
                bound_strengthened += 1
            if (final_new_lower_bounds[j] > lb[j]) and (final_new_lower_bounds[j] < ub[j]):
                lb[j] = final_new_lower_bounds[j]
                bound_strengthened += 1
        
        save_json(A, b, c,final_new_lower_bounds, final_new_upper_bounds, of_sense, cons_sense, "model_matrices.json", co, variable_names) 

        return final_new_lower_bounds, final_new_upper_bounds


def standard_form(model):
    """
    Converts a given Gurobi model into its standard form.

    In the standard form:
    - All variables are non-negative.
    - All constraints are equalities.
    - The objective is to be minimized.

    Args:
    - model: The Gurobi model to be converted.

    Returns:
    - standard_model: The converted model in standard form.
    - track_elements: A dictionary tracking the changes made to variables and constraints.
    """

    # Initialize tracking dictionaries for variables and constraints
    track_elements = {'variables': {}, 'constraints': {}}

    # Clone the model to avoid changing the original
    standard_model = model.copy()

    # Set the model sense to minimization
    if standard_model.ModelSense != 1:
        # Multiply the objective function by -1 to switch from maximization to minimization
        standard_model.setObjective(-1 * standard_model.getObjective(), gp.GRB.MINIMIZE)
        standard_model.ModelSense = 1  # Set the model sense to minimization

    standard_model.update()

    # Create a mapping from old variables to their replacements
    var_replacements = {}

    # Ensure all variables are non-negative
    for var in standard_model.getVars():
        lower_bound = var.LB
        upper_bound = var.UB

        if lower_bound == 0 and upper_bound == gp.GRB.INFINITY:
            # Variable is already non-negative
            track_elements['variables'][var.VarName] = 'original_non_negative'
            continue

        # Replace variable with non-negative variables
        if lower_bound == -gp.GRB.INFINITY and upper_bound == gp.GRB.INFINITY:
            # Variable is unrestricted in sign
            pos_var = standard_model.addVar(lb=0, name=f"{var.VarName}_pos")
            neg_var = standard_model.addVar(lb=0, name=f"{var.VarName}_neg")
            var_replacements[var] = (pos_var, neg_var)

            # Replace var in constraints and objective
            track_elements['variables'][var.VarName] = 'replaced_with_pos_neg'
        else:
            # Variable has finite bounds
            shift = lower_bound if lower_bound != -gp.GRB.INFINITY else 0
            scale = 1
            if upper_bound != gp.GRB.INFINITY and lower_bound != -gp.GRB.INFINITY:
                scale = upper_bound - lower_bound

            new_var = standard_model.addVar(lb=0, name=f"{var.VarName}_nonneg")
            var_replacements[var] = (new_var, shift, scale)
            track_elements['variables'][var.VarName] = 'replaced_with_non_negative'

    standard_model.update()

    # Replace variables in constraints and objective
    for constr in standard_model.getConstrs():
        expr = standard_model.getRow(constr)
        new_expr = gp.LinExpr()
        for i in range(expr.size()):
            coeff = expr.getCoeff(i)
            var = expr.getVar(i)
            if var in var_replacements:
                replacement = var_replacements[var]
                if isinstance(replacement, tuple) and len(replacement) == 2:
                    # Unrestricted variable replaced with two non-negative variables
                    pos_var, neg_var = replacement
                    new_expr.addTerms(coeff, pos_var)
                    new_expr.addTerms(-coeff, neg_var)
                else:
                    # Variable with finite bounds replaced
                    new_var, shift, scale = replacement
                    new_expr.addTerms(coeff * scale, new_var)
                    constr.RHS -= coeff * shift
            else:
                new_expr.addTerms(coeff, var)
        standard_model.remove(constr)
        standard_model.addConstr(new_expr, constr.Sense, constr.RHS, name=constr.ConstrName)

    # Replace variables in the objective function
    obj = standard_model.getObjective()
    new_obj = gp.LinExpr()
    for i in range(obj.size()):
        coeff = obj.getCoeff(i)
        var = obj.getVar(i)
        if var in var_replacements:
            replacement = var_replacements[var]
            if isinstance(replacement, tuple) and len(replacement) == 2:
                # Unrestricted variable replaced with two non-negative variables
                pos_var, neg_var = replacement
                new_obj.addTerms(coeff, pos_var)
                new_obj.addTerms(-coeff, neg_var)
            else:
                # Variable with finite bounds replaced
                new_var, shift, scale = replacement
                new_obj.addTerms(coeff * scale, new_var)
                obj.addConstant(coeff * shift)
        else:
            new_obj.addTerms(coeff, var)
    standard_model.setObjective(new_obj, gp.GRB.MINIMIZE)

    # Now, handle inequality constraints as in your original code
    # (Convert them to equalities by adding slack or surplus variables)

    # [Include your existing code for converting inequalities to equalities here]

    standard_model.update()

    return standard_model, track_elements


def standard_form1(model):
    print("Converting the model to standard form...")
    # Step 1: Ensure the model is in minimization form
    if model.ModelSense != 1:
        model.setObjective(-1 * model.getObjective(), GRB.MINIMIZE)
        model.ModelSense = 1  # Set the model sense to minimization

    # Step 2: Ensure all variables are non-negative
    vars_to_remove = []
    for var in model.getVars():
        print(f"Processing variable {var.VarName}")
        if var.LB != 0 or var.UB != GRB.INFINITY:
            # Introduce two non-negative variables (positive and negative part)
            pos_var = model.addVar(lb=0, name=f"{var.VarName}_pos")
            neg_var = model.addVar(lb=0, name=f"{var.VarName}_neg")
            
            # Add constraints to enforce original bounds
            if var.UB < GRB.INFINITY:
                print(f"Adding upper bound constraint for {var.VarName}")
                model.addConstr(pos_var - neg_var <= var.UB, name=f"{var.VarName}_UB")
            if var.LB > -GRB.INFINITY:
                print(f"Adding lower bound constraint for {var.VarName}")
                model.addConstr(pos_var - neg_var >= var.LB, name=f"{var.VarName}_LB")

            # Replace 'var' in constraints
            for constr in model.getConstrs():
                coeff = model.getCoeff(constr, var)
                if coeff != 0:
                    model.chgCoeff(constr, pos_var, coeff)
                    model.chgCoeff(constr, neg_var, -coeff)
                    model.chgCoeff(constr, var, 0)

            # Replace 'var' in the objective
            obj_coeff = var.Obj
            if obj_coeff != 0:  # Ensure it's part of the objective function
                model.setObjective(model.getObjective() + obj_coeff * (pos_var - neg_var), GRB.MINIMIZE)

            # Mark the original variable for removal
            vars_to_remove.append(var)

        # Remove the marked variables after all replacements are done
        model.update()
        for var in vars_to_remove:
            model.remove(var)
        model.update()

    # Step 3: Transform all constraints into equalities by introducing slack/surplus variables
    for constr in model.getConstrs():
        sense = constr.Sense
        if sense != GRB.EQUAL:
            # Add slack or surplus variable based on constraint type
            slack_var = model.addVar(lb=0, name=f"slack_{constr.ConstrName}")
            model.update()

            # Add slack or surplus depending on the sense of the constraint
            if sense == GRB.LESS_EQUAL:
                model.chgCoeff(constr, slack_var, 1)
            elif sense == GRB.GREATER_EQUAL:
                model.chgCoeff(constr, slack_var, -1)

            # Set the constraint to equality after adding slack/surplus
            constr.Sense = GRB.EQUAL

        # Exclude the objective function variable from being part of the constraints
        if constr.ConstrName == 'obj':
            continue  # Skip any handling for the objective constraint

    model.update()
    # Modify the objective coefficients or redefine the objective
    for v in model.getVars():
        v.obj = 1.0  # Set a new objective coefficient for each variable
    model.setObjective(gp.quicksum(v for v in model.getVars()), GRB.MINIMIZE)  # Redefine the objective function
    
    return model

                        #     variable_coeffs[linked_var] = coefficient
    # Return the detected auxiliary objective variable and its linked variable coefficients

# def modify_mps_objective(file_path, output_path):
#     """
#     Modify an MPS file to replace an auxiliary variable in the objective function with
#     a linear combination of other variables, and change the sign of the coefficients.

#     Parameters:
#     - file_path: Path to the input MPS file.
#     - output_path: Path to the output modified MPS file.
#     """
#     # Read the original MPS file
#     with open(file_path, 'r') as file:
#         mps_content = file.readlines()

#     # Detect the objective variable and its coefficients
#     objective_var, variable_coeffs, equation = detect_objective_and_coefficients(file_path)

#     del variable_coeffs[objective_var]

#     # If the objective variable is found, modify the coefficients
#     if not objective_var or not variable_coeffs:
#         print("No auxiliary objective variable or coefficients detected.")
#         return

#     modified_mps_content = []
#     in_columns_section = False
#     objective = 0

#     for line in mps_content:
#         if 'maximizing' in line:
#             objective = -1
#         if 'minimizing' in line:
#             objective = 1
#         # Start modifying the COLUMNS section
#         if line.startswith('COLUMNS'):
#             modified_mps_content.append(line)
#             in_columns_section = True
#             continue

#         # In the COLUMNS section, replace the auxiliary objective variable with the real variables
#         if in_columns_section:
#             parts = line.split()
#             if len(parts) >= 3:
#                 var_name = parts[0]
#                 row_name = parts[1]
#                 coefficient = parts[2]

#                 # If the coefficient matches the detected variable, we modify the row to 'obj'
#                 if var_name in variable_coeffs and row_name == equation:

#                     if float(coefficient) == variable_coeffs[var_name]:
#                         # Replace the line with the updated row name and sign-changed coefficient
#                         if objective == 1:
#                             modified_mps_content.append(f"    {var_name}    obj    {-variable_coeffs[var_name]}\n")
#                             print(f"Modified line: {var_name}    obj    {-variable_coeffs[var_name]}")
#                         elif objective == -1:
#                             modified_mps_content.append(f"    {var_name}    obj    {variable_coeffs[var_name]}\n")
#                             #print(f"Modified line: {var_name}    obj    {variable_coeffs[var_name]}")
#                     else: 
#                         print(objective_var, var_name)
            
#                         # Append other lines as they are
#                         modified_mps_content.append(line)
#                 else:
#                     if var_name == objective_var:
#                         continue
#                     else:
#                         # Append other lines as they are
#                         modified_mps_content.append(line)
#                     # Append other lines as they are
#                     #modified_mps_content.append(line)
#             else:
#                 # Handle lines that don't contain valid parts (just append them as is)
#                 modified_mps_content.append(line)
#             # if line.startswith('RHS') or line.startswith('BOUNDS'):
#             #     in_columns_section = False
#         else:
#             if equation in line:
#                 print('Entro')
#                 print(line)
#                 # Replace the objective variable with the real variables
#                 continue
#             # Append lines outside the COLUMNS section as they are
#             modified_mps_content.append(line)
            

#     # Save the modified MPS file
#     with open(output_path, 'w') as modified_file:
#         modified_file.writelines(modified_mps_content)
#     print(f"Modified MPS file saved to: {output_path}")

def sensitivity_analysis(modelo, datos):
    if modelo in datos:
        modelo_datos = datos[modelo]
        modelo_primal = modelo_datos.get('primal', {})
        modelo_dual = modelo_datos.get('dual', {})
        modelo_pr_eps = modelo_primal.get('epsilon', [])
        modelo_pr_of = modelo_primal.get('objective_function', [])
        modelo_pr_dv = modelo_primal.get('decision_variables', [])
        modelo_pr_ci = modelo_primal.get('changed_indices', [])
        modelo_pr_cv = modelo_primal.get('constraint_violation', [])
        modelo_pr_ofod = modelo_primal.get('of_original_decision', [])
        modelo_pr_time = modelo_primal.get('execution_time', [])
        modelo_du_dv = modelo_dual.get('decision_variables', [])
        
        # Cálculo de degradación de la función objetivo original
        divisor = modelo_pr_of[0] if modelo_pr_of else None
        modelo_pr_of_pu = [elemento / divisor for elemento in modelo_pr_of] if divisor else None
        
        # Cálculo de degradación de la función objetivo original
        divisor1 = modelo_pr_ofod[0] if modelo_pr_ofod else None
        modelo_pr_ofod_pu = [elemento / divisor1 for elemento in modelo_pr_ofod] if divisor1 else None
        
        ## Cálculo de INFEASIBILITY INDEX
        cifra_referencia = 1e-6
        modelo_pr_cv_sin_nan = remove_nan_sublists(modelo_pr_cv)
        modelo_pr_cv_filtrado = set_values_below_threshold_to_zero(modelo_pr_cv_sin_nan, cifra_referencia)
        producto_cv_vd = multiply_matrices(modelo_pr_cv_filtrado, modelo_du_dv)
        suma_producto = sum_sublists(producto_cv_vd)
        infeasiblity_index = [x / abs(modelo_pr_ofod[0]) for x in suma_producto]
        
        ## Cálculo de PROBLEM COMPLEXITY
        ## Cálculo de la media de la constraint violations
        modelo_pr_cv_medias = calculate_means(modelo_pr_cv)
        
        ## Cálculo de los índices cambiados a 0 de la matriz A
        acumulado_modelo_ci = calculate_lengths(modelo_pr_ci)
        # convert_late_zeros_to_nan(acumulado_modelo_ci) # ¿Qué es esta función?
        elementos_A_totales = len(modelo_pr_dv) * len(modelo_du_dv)
        
        ## Ahora calculamos el número de no 0s en la matriz A, para cada valor de epsilon.
        ## Esto lo podemos calcular mediante el número de indices que cambian en cada nivel de epsilon
        elementos_A_que_se_hacen_0_pu = [x / elementos_A_totales for x in acumulado_modelo_ci]
        complexity_problem = [1 - x for x in elementos_A_que_se_hacen_0_pu]
        
        suma_objfunc_unfeasiblity = [a + b for a, b in zip(modelo_pr_ofod_pu, infeasiblity_index)]
        
        ## PROBLEM COMPLEXITY, con el tiempo de ejecución (comentado en el código original)
        # modelo_pr_time1 = modelo_pr_time[1:]
        # #Convertir los tiempos a números flotantes
        # modelo_pr_time1 = [float(tiempo.split(':')[2]) for tiempo in modelo_pr_time1]
        # divisor = modelo_pr_time1[0] if modelo_pr_time1 else None
        # modelo_pr_time_pu = [elemento / divisor for elemento in modelo_pr_time1] if divisor else None
        # complexity_problem = modelo_pr_time_pu
        
        ## GRÁFICAS
        titulo1 = (modelo + " Objective function degradation")
        titulo2 = (modelo + " Infeasibility evolution")
        titulo3 = (modelo + " Complexity evolution")
        objective_function = "Objective function"
        complexity = "Complexity"
        infeasibility = "Infeasibility"
        
        # plot1(modelo_pr_eps, modelo_pr_of_pu, titulo1, objective_function)
        # plot1(modelo_pr_eps, infeasiblity_index, titulo2, infeasibility)
        # plot1(modelo_pr_eps, complexity_problem, titulo3, complexity)
        
        plot_subplots(modelo_pr_eps, modelo_pr_of_pu, infeasiblity_index, complexity_problem, titulo1, titulo2, titulo3)
        return
    else:
        print(f"El modelo '{modelo}' no se encontró en los datos proporcionados.")
        return None
