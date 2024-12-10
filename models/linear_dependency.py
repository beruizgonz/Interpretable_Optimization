import numpy as np
 
# # Definir la matriz A con las filas adicionales
A = np.array([
    [2, 1, 0],
    [2, 1, 1],
    [4, 2, 0],
    [6, 3, 2],
    [0, 0, 1],
    [2, 1, 0]
 ], dtype=float)
 
# # Definir la tolerancia
TOLERANCIA = 1e-2
rank = np.linalg.matrix_rank(A)

# Función para determinar las filas base con tolerancia
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
                print(coeficientes)
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
 
# Obtener las filas base y sus índices
filas_base, indices_base = obtener_filas_base(A, TOLERANCIA)
 
print("Filas base seleccionadas (índices comenzando en 0):", indices_base)
print("Filas base:")
for i, fila in zip(indices_base, filas_base):
    print(f"Fila {i+1}: {fila}")
 
# Función para expresar cada fila como combinación lineal de las filas base
def expresar_combinaciones(matriz, filas_base):
    base_matrix = np.vstack(filas_base).T  # Cada columna es una fila base
    combinaciones = []
    for i, fila in enumerate(matriz):
        # Resolver base_matrix * coef = fila
        coeficientes, residuals, rango, _ = np.linalg.lstsq(base_matrix, fila, rcond=None)
        combinaciones.append(coeficientes)
    return combinaciones
 
# Obtener las combinaciones lineales
combinaciones = expresar_combinaciones(A, filas_base)
 
print("\nExpresiones como combinaciones lineales de las filas base:")
for i, coef in enumerate(combinaciones):
    expresion = f"Fila {i+1} = "
    terminos = []
    for j, c in enumerate(coef):
        if abs(c) > 1e-10:  # Ignorar coeficientes cercanos a cero
            terminos.append(f"{c:.2f} * Fila {indices_base[j]+1}")
    expresion += " + ".join(terminos) if terminos else "0"
    print(expresion)

# Define the matrix
# A = np.array([
#     [1, 2, 3],
#     [2, 4, 6],
#     [3, 6, 9]
# ])

# Convert to sympy Matrix for RREF
# A_sym = Matrix(A)
# rref, pivot_columns = A_sym.rref()

# # Display RREF and identify dependent rows
# print("RREF of the matrix:")
# print(np.array(rref))

# print("Pivot columns:", pivot_columns)
# dependent_rows = [i for i in range(A.shape[0]) if i not in pivot_columns]
# print("Dependent rows:", dependent_rows)

    