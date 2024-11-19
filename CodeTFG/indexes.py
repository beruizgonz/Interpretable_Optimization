import json
from auxiliary_functions import *
from statistics import *
import os 
import numpy as np

def optimality_index(model, data):
        if model in data:
            modelo_datos = data[model]
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
        modelo_pr_of_pu = [x * 100 for x in modelo_pr_of_pu]

        # Cálculo del índice de optimalidad
        # optimal_index = [x / divisor for x in modelo_pr_of]
        # optimal_index = [x * 100 for x in optimal_index]

        return modelo_pr_of_pu

def infeasibility_index(model, data): 
    if model in data:
        modelo_datos = data[model]
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
    
    cifra_referencia = 1e-6
    modelo_pr_cv_sin_nan = remove_nan_sublists(modelo_pr_cv)
    modelo_pr_cv_filtrado = set_values_below_threshold_to_zero(modelo_pr_cv_sin_nan, cifra_referencia)
    producto_cv_vd = multiply_matrices(modelo_pr_cv_filtrado, modelo_du_dv)
    suma_producto = sum_sublists(producto_cv_vd)
    infeasiblity_index = [abs(x) / abs(modelo_pr_ofod[0]) for x in suma_producto]

    return infeasiblity_index

def complexity_index(model, data):
    if model in data:
        modelo_datos = data[model]
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
        rows_pr_changed = modelo_primal.get('rows_changed', [])
        columns_pr_changed = modelo_primal.get('columns_changed', [])
        total_non_zeros = modelo_primal.get('non_zeros', [])

    modelo_pr_cv_medias = calculate_means(modelo_pr_cv)

    constraints = len(modelo_pr_cv[0])
    variables = len(modelo_pr_dv[0])
    total_elements_A = total_non_zeros
    

    # Cosntrucción de la matriz A
    A_matrix = np.zeros((constraints, variables))
    # Rellenamos la matriz A con los inices cambiados a 0
    complexity_index1 = []
    rows = []
    columns = []
    for i in modelo_pr_ci:
        if i != [] and i is not None:
            for j in i:
                A_matrix[j[0]][j[1]] = 1
        
        counts_ones = np.count_nonzero(A_matrix)
        # counts the rows with all ones
        counts_ones_rows = np.count_nonzero(np.all(A_matrix, axis=1))
        counts_ones_columns = np.count_nonzero(np.all(A_matrix, axis=0))  
        complexity_index1.append(1 - counts_ones / total_elements_A)


    for row in rows_pr_changed:
        if row is None:
            complexity_rows = 1

        else: 
            if len(row) > 0:
                print('Row:', len(row))
            complexity_rows = 1 - len(row) / constraints
        rows.append(complexity_rows)

    for col in columns_pr_changed:
        if col is None:
            complexity_columns = 1
        else:
            if len(col) > 0:
                print('Column:', len(col))
            complexity_columns = 1 - len(col) / variables
     
       
        
        columns.append(complexity_columns)
        #print(total_elements_A)
            #print(total_elements_A)
    ## Cálculo de los índices cambiados a 0 de la matriz A
    acumulado_modelo_ci = calculate_lengths(modelo_pr_ci)
    # convert_late_zeros_to_nan(acumulado_modelo_ci) # ¿Qué es esta función?
    elementos_A_totales = len(modelo_pr_dv) * len(modelo_du_dv)
    #print(elementos_A_totales)
    
    ## Ahora calculamos el número de no 0s en la matriz A, para cada valor de epsilon.
    ## Esto lo podemos calcular mediante el número de indices que cambian en cada nivel de epsilon
    elementos_A_que_se_hacen_0_pu = [x / elementos_A_totales for x in acumulado_modelo_ci]
    complexity_index = [1 - x for x in elementos_A_que_se_hacen_0_pu]
    
    # suma_objfunc_unfeasiblity = [a + b for a, b in zip(modelo_pr_ofod_pu, infeasiblity_index)]
    return [complexity_index1, rows, columns]

def execution_time_index(model, data):
    if model in data:
        modelo_datos = data[model]
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

    ## PROBLEM COMPLEXITY, con el tiempo de ejecución (comentado en el código original)
    modelo_pr_time1 = modelo_pr_time[1:]
    #Convertir los tiempos a números flotantes
    modelo_pr_time1 = [float(tiempo.split(':')[2]) for tiempo in modelo_pr_time1]
    divisor = modelo_pr_time1[0] if modelo_pr_time1 else None
    modelo_pr_time_pu = [elemento / divisor for elemento in modelo_pr_time1] if divisor else None
    complexity_problem = modelo_pr_time_pu

    
     
        