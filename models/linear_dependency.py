import numpy as np
import os 
from scipy.sparse import csr_matrix, vstack
from scipy.linalg import qr
from scipy.sparse.linalg import lsqr
import gurobipy as gp

from utils_models.utils_functions import * 

# PATH TO THE DATA
project_root = os.path.dirname(os.getcwd())
model_path = os.path.join(project_root, 'data/GAMS_library', 'DECOMP.mps')
GAMS_path = os.path.join(project_root, 'data/GAMS_library')
real_data_path = os.path.join(project_root, 'data/real_data')
bounds_path = os.path.join(project_root, 'data/bounds_trace')

GAMS_path_modified = os.path.join(project_root, 'data/GAMS_library_modified')
model_path_modified = os.path.join(GAMS_path_modified, 'FAWLEY.mps')
real_model_path = os.path.join(real_data_path,  'openTEPES_9n_2030_sc01_st1.mps')

 # OTHER DATA
 # DEFINE A MATRIX A THAT IS AN EXAMPLE OF A MATRIX WITH LINEAR DEPENDENCY
A = np.array([
    [2, 1, 0],
    [2, 1, 1],
    [4, 2, 0],
    [6, 3, 2],
    [0, 0, 1],
    [2, 1.0001, 0]
 ], dtype=float)
 
# TOLERANCE FOR THE LINEAR DEPENDENCY
TOLERANCE = 1e-2

###############################################################################
# 1) ROW-BASED ITERATIVE METHOD
###############################################################################
def obtener_filas_base(matriz, tolerancia):
    filas_base = []
    indices_base = []
    for i, fila in enumerate(matriz):
        if len(filas_base) == 0:
            filas_base.append(fila)
            indices_base.append(i)
        else:
            # Crear una matriz con las filas base actuales como columnas
            base_matrix = np.vstack(filas_base).T  # Cada columna es una fila base
            try:
                # Resolver el sistema base_matrix * coef = fila
                coeficientes, residuals, rango, _ = np.linalg.lstsq(base_matrix, fila, rcond=None)
                if residuals.size > 0:
                    residual_norm = np.sqrt(residuals[0])
                else:
                    # Calcular la norma del residual manualmente si residuals está vacío
                    residual = fila - base_matrix @ coeficientes
                    residual_norm = np.linalg.norm(residual)
                # Si la norma del residual es mayor que la tolerancia, agregar la fila a la base
                if residual_norm > tolerancia:
                    filas_base.append(fila)
                    indices_base.append(i)
            except np.linalg.LinAlgError:
                filas_base.append(fila)
                indices_base.append(i)
    return filas_base, indices_base

def find_sparse_independent_rows_iterative(A_csr, tol=1e-2):
    rows_base = []
    indices_base = []

    # We'll store the basis rows in some form we can solve with quickly:
    # e.g., stacking them in a dense or sparse structure. 
    # (But fully dense might again be huge; see note below about memory.)
    basis_data = []

    for i in range(A_csr.shape[0]):
        row_i = A_csr.getrow(i)
        if not rows_base:
            rows_base.append(row_i)
            indices_base.append(i)
            basis_data.append(row_i.toarray().ravel())
        else:
            # Build a matrix shape (num_cols, len(rows_base)) in dense or sparse
            base_array = np.array(basis_data).T  # shape (n_cols, #base_rows)
            # We solve base_array * x = row_i_dense in least squares
            row_i_dense = row_i.toarray().ravel()

            sol = lsqr(base_array, row_i_dense, atol=1e-12, btol=1e-12)
            residual_norm = sol[3]  # 2-norm of residual from LSQR
            if residual_norm > tol:
                rows_base.append(row_i)
                indices_base.append(i)
                basis_data.append(row_i_dense)

    return indices_base


###############################################################################
# 2) ROW-BASED PIVOTED QR (NOT SPARSE MATRIX)
##############################################################################
def rows_basis_qr_descomposition(new_A, epsilon=1e-2):     
    print(new_A.shape)  
    Q, R, P = qr(new_A.T, pivoting=True)  # shape of R: (n_rows, n_rows)
    diag_R = np.abs(np.diag(R))
    rank = np.sum(diag_R > epsilon)

    # The first `rank` elements in P are the indices of the linearly independent rows.
    pivot_rows = P[:rank]

    # Zero out the rows that are not pivot rows
    # (This is analogous to what you do in your code.)
    # Make a copy or modify in place as needed
    A_out = new_A.copy()
    for i in range(A_out.shape[0]):
        if i not in pivot_rows:
            A_out[i, :] = 0
            #b_out[i, 0] = 0

    # Convert back to sparse if needed
    A_out_csr = csr_matrix(A_out)

    # Return the linearly independent row indices (pivot_rows) and the modified A, b
    return pivot_rows, A_out_csr

###############################################################################
# 2) COLUMNS-BASED PIVOTED QR
##############################################################################
def columns_basis_qr_descomposition(A, epsilon):
    """
    Find a basis for the columns of A using pivoted QR, up to the tolerance epsilon.
    
    Parameters
    ----------
    A : np.ndarray or scipy.sparse.csr_matrix
        The input matrix of shape (m, n).
    epsilon : float
        Tolerance to decide which pivots (diagonal entries of R) are considered significant.
    
    Returns
    -------
    pivot_columns : np.ndarray
        1D array of column indices in A that form a basis (linearly independent set).
    A_out_csr : scipy.sparse.csr_matrix
        A copy of A (in sparse CSR format) where columns not in pivot_columns have been zeroed out.
    """
    # If A is sparse, convert to dense for QR
    if not isinstance(A, np.ndarray):
        A = A.toarray()

    # Perform pivoted QR on A
    # Q has shape (m, m), R has shape (m, n), P is a permutation of [0..n-1]
    Q, R, P = qr(A, pivoting=True)

    # The diagonal of R corresponds to the pivot magnitudes for each column 
    # in the pivoted order. We compare them to epsilon to determine rank.
    diag_R = np.abs(np.diag(R))
    rank = np.sum(diag_R > epsilon)

    # The first 'rank' columns in P are the pivot columns that are linearly independent
    pivot_columns = P[:rank]

    # Zero out the *non-pivot* columns in a copy of A
    A_out = A.copy()
    all_columns = set(range(A.shape[1]))
    pivot_set = set(pivot_columns)

    for col_idx in all_columns - pivot_set:
        A_out[:, col_idx] = 0.0

    # Convert back to sparse (if desired)
    A_out_csr = csr_matrix(A_out)
    return pivot_columns, A_out_csr

# Función para expresar cada fila como combinación lineal de las filas base
def combinations_rows(matriz, filas_base):
    base_matrix = np.vstack(filas_base).T  # Cada columna es una fila base
    combinaciones = []
    for i, fila in enumerate(matriz):
        # Resolver base_matrix * coef = fila
        coeficientes, residuals, rango, _ = np.linalg.lstsq(base_matrix, fila, rcond=None)
        combinaciones.append(coeficientes)
    return combinaciones

def combinations_columns(matriz, columnas_base):
    """
    Expresa cada columna de 'matriz' como una combinación lineal
    de las columnas en 'columnas_base'.
    
    Parámetros
    ----------
    matriz : np.ndarray
        Matriz de dimensiones (m, n). Vamos a expresar cada una
        de sus n columnas.
    columnas_base : list or np.ndarray
        Conjunto de columnas base. Puede ser:
          - Una lista de vectores 1D (cada uno de dimensión m).
          - Una matriz 2D de dimensión (m, r) donde cada columna
            ya es una columna base.
    
    Devuelve
    --------
    combinaciones : list of np.ndarray
        Lista de coeficientes. El elemento combinaciones[j] es el
        vector de coeficientes que permite expresar la columna j
        de 'matriz' como combinación lineal de 'columnas_base'.
    """
    # Asegurarnos de que columnas_base sea una matriz 2D (m, r)
    if isinstance(columnas_base, list):
        # Suponemos que cada elemento es un vector de dimensión m
        base_matrix = np.column_stack(columnas_base)  # (m, r)
    else:
        # Suponemos que ya es (m, r)
        base_matrix = columnas_base

    combinaciones = []
    
    # Para cada columna j de 'matriz'
    for j in range(matriz.shape[1]):
        col_j = matriz[:, j]  # Extraemos la columna j
        # Resolvemos base_matrix * coef = col_j
        coeficientes, residuals, rango, _ = np.linalg.lstsq(base_matrix, col_j, rcond=None)
        combinaciones.append(coeficientes)
    
    return combinaciones

def print_combinations_rows(combinaciones, filas_base, indices_base):
    print("\nExpresiones como combinaciones lineales de las filas base:")
    for i, coef in enumerate(combinaciones):
        expresion = f"Fila {i+1} = "
        terminos = []
        for j, c in enumerate(coef):
            if abs(c) > 1e-10:  # Ignorar coeficientes cercanos a cero
                terminos.append(f"{c:.2f} * Fila {indices_base[j]+1}")
        expresion += " + ".join(terminos) if terminos else "0"
        print(expresion)

def print_combinations_columns(combinaciones, columnas_base, indices_base):
    print("\nExpresiones como combinaciones lineales de las columnas base:")
    for i, coef in enumerate(combinaciones):
        expresion = f"Columna {i+1} = "
        terminos = []
        for j, c in enumerate(coef):
            if abs(c) > 1e-10:  # Ignorar coeficientes cercanos a cero
                terminos.append(f"{c:.2f} * Columna {indices_base[j]+1}")
        expresion += " + ".join(terminos) if terminos else "0"
        print(expresion)
 
if __name__ == '__main__':
    # filas_base, indices_base = obtener_filas_base(A, TOLERANCE)
    # comb = combinations_rows(A, filas_base)
    # print_combinations_rows(comb, filas_base, indices_base)

    model = gp.read(real_model_path)
    A, b, c, co, lb, ub, of_sense, cons_senses, variable_names = get_model_matrices(model)
    b = np.array(b)
    A_amplied = vstack([A.T, b.reshape(1, -1)]).T
    A_amplied = csr_matrix(A_amplied)
    index_basis = find_sparse_independent_rows_iterative(A_amplied, tol=0.021599999)
    # pivot_rows, A_out_csr = rows_basis_qr_descomposition(A_amplied, epsilon=0.021599999)
    # comb = combinations_rows(A_amplied, A_out_csr.toarray()[pivot_rows])
    # print_combinations_rows(comb, pivot_rows, pivot_rows)
    # # Check that the combination computed is correct
    # print("Check that the combination computed is correct")
    # comb = np.array(comb)
    # print(comb.shape)
    # print(A_amplied.shape)
    # print(np.allclose(A_amplied, comb @ A_out_csr.toarray()[pivot_rows], atol=0.27))