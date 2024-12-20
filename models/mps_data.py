import os

def detect_objective_and_coefficients(file_path):
    """
    Detect the auxiliary objective variable and the coefficients of the actual variables linked to it.

    Parameters:
    - file_path: Path to the MPS file.

    Returns:
    - objective_var: The detected auxiliary variable used in the objective (e.g., 'x7').
    - variable_coeffs: A dictionary where keys are variable names (e.g., 'x1') and values are their coefficients.
    """
    in_columns_section = False
    objective_var = None
    equation = None
    variable_coeffs = {}
    
    with open(file_path, 'r') as file:
        mps_content = file.readlines()

    # Step 1: Detect the auxiliary objective variable (the one linked to 'obj')
    for line in mps_content:
        # Detect the start of the COLUMNS section
        if line.startswith('COLUMNS'):
            in_columns_section = True
            #continue
        
        if in_columns_section:
            
            parts = line.split()
            if len(parts) >= 3:
                var_name = parts[0]  # Variable name (e.g., x1, x2, etc.)
                row_name = parts[1]  # 'obj' or any constraint name
                
                if row_name.lower() == 'obj':
                    # Capture the first variable linked to 'obj' as the objective_var
                    if not objective_var:
                        objective_var = var_name
                    else:
                        # Capture other variables linked to 'obj' and their coefficients
                        variable_coeffs[var_name] = float(parts[2])

    for line in mps_content:
        if line.startswith('COLUMNS'):
            in_columns_section = True
            continue

        # if line.startswith('RHS'):
        #     in_columns_section = False
        #     break

        if in_columns_section:
            parts = line.split()
            if len(parts) >= 3:
               if parts[0] == objective_var:
                   variable_name = parts[1]
                   equation = variable_name
    # # Step 2: Detect variables linked to the auxiliary objective variable (equation form)
    in_columns_section = False
    if variable_name:
        for line in mps_content:
            if line.startswith('COLUMNS'):
                in_columns_section = True
                continue
            
            if in_columns_section:

                # End when find RHS section
                if line.startswith('RHS'):
                    break
                parts = line.split()
                if len(parts) >= 3:
                    var_name = parts[0]  # Variable name (e.g., x1, x2, etc.)
                    linked_var = parts[1]  # The constraint or row name
                    coefficient = float(parts[2])  # Coefficient of the variable
                    
                    # If the vareiable is linked to the auxiliary objective variable, store its coefficient
                    if linked_var == variable_name:
                        #if linked_var in variable_coeffs:
                            #If the variable is already in the dictionary, sum up the coefficients
                        variable_coeffs[var_name] = coefficient
                        # else:
    return objective_var, variable_coeffs, equation

def detect_objective_and_coefficients_new(file_path):
    """
    Detect the auxiliary objective variable and the coefficients of the actual variables linked to it.

    Parameters:
    - file_path: Path to the MPS file.

    Returns:
    - objective_var: The detected auxiliary variable used in the objective (e.g., 'x7').
    - variable_coeffs: A dictionary where keys are variable names (e.g., 'x1') and values are their coefficients.
    - equation: The row name (constraint) representing the actual objective function.
    """
    in_columns_section = False
    in_rhs_section = False
    objective_var = None
    equation = None
    variable_coeffs = {}

    with open(file_path, 'r') as file:
        mps_content = file.readlines()

    # Step 1: Detect the auxiliary objective variable in the COLUMNS section
    for line in mps_content:
        # Detect the start of the COLUMNS section
        if line.startswith('COLUMNS'):
            in_columns_section = True
            continue

        if in_columns_section:
            # Exit the COLUMNS section when RHS is encountered
            if line.startswith('RHS'):
                in_columns_section = False
                break

            parts = line.split()
            if len(parts) >= 3:
                var_name = parts[0]  # Variable name (e.g., x1, x2, etc.)
                row_name = parts[1]  # Row name (e.g., 'obj' or constraint name)

                # Identify the auxiliary objective variable linked to 'obj'
                if row_name.lower() == 'obj':
                    if not objective_var:
                        objective_var = var_name
                elif var_name == objective_var:
                    # Detect the row representing the actual objective (equation)
                    equation = row_name

    # Step 2: Collect coefficients of variables linked to the auxiliary objective variable
    if equation:
        in_columns_section = False
        for line in mps_content:
            if line.startswith('COLUMNS'):
                in_columns_section = True
                continue

            if in_columns_section:
                # End when RHS section is found
                if line.startswith('RHS'):
                    break

                parts = line.split()
                if len(parts) >= 3:
                    var_name = parts[0]  # Variable name (e.g., x1, x2, etc.)
                    row_name = parts[1]  # Constraint or row name
                    coefficient = float(parts[2])  # Coefficient of the variable

                    # If the variable is linked to the actual objective row, store its coefficient
                    if row_name == equation:
                        variable_coeffs[var_name] = coefficient

    return objective_var, variable_coeffs, equation

def modify_mps_objective(file_path, output_path):
    """
    Modify an MPS file to replace an auxiliary variable in the objective function with
    a linear combination of other variables, and change the sign of the coefficients.

    Parameters:
    - file_path: Path to the input MPS file.
    - output_path: Path to the output modified MPS file.
    """
    # Read the original MPS file
    with open(file_path, 'r') as file:
        mps_content = file.readlines()

    # Detect the objective variable and its coefficients
    objective_var, variable_coeffs, equation = detect_objective_and_coefficients(file_path)
    print('Objective variable:', objective_var)
    del variable_coeffs[objective_var]

    # If the objective variable is found, modify the coefficients
    if not objective_var or not variable_coeffs:
        print("No auxiliary objective variable or coefficients detected.")
        return

    modified_mps_content = []
    in_columns_section = False
    after_columns_section = False
    objective = 0
    print(f"Objective variable: {objective_var}")
    print(f"Equation: {equation}")  

    for line in mps_content:

        if 'maximizing' in line:
            objective = -1
        if 'minimizing' in line:
            objective = 1
        # Start modifying the COLUMNS section
        if line.startswith('COLUMNS'):
            modified_mps_content.append(line)
            in_columns_section = True
            
        if line.startswith('RHS') or line.startswith('BOUNDS'):
            in_columns_section = False
            after_columns_section = True

        # In the COLUMNS section, replace the auxiliary objective variable with the real variables
        if in_columns_section:
            parts = line.split()
            if len(parts) >= 3:
                var_name = parts[0]
                row_name = parts[1]
                coefficient = parts[2]

                # If the coefficient matches the detected variable, we modify the row to 'obj'
                if var_name in variable_coeffs and row_name == equation:

                    if float(coefficient) == variable_coeffs[var_name]:
                        # Replace the line with the updated row name and sign-changed coefficient
                        if objective == 1:
                            modified_mps_content.append(f"    {var_name}    obj    {-variable_coeffs[var_name]}\n")
                            print(f"Modified line: {var_name}    obj    {-variable_coeffs[var_name]}")
                        elif objective == -1:
                            modified_mps_content.append(f"    {var_name}    obj    {variable_coeffs[var_name]}\n")
                            #print(f"Modified line: {var_name}    obj    {variable_coeffs[var_name]}")
                    else: 
                        print(objective_var, var_name)
            
                        # Append other lines as they are
                        modified_mps_content.append(line)

                elif equation == row_name and var_name == objective_var:
                    print(line)
                    continue
                elif var_name == objective_var and row_name == 'obj':
                    print(line)
                    continue
                else:
                    if row_name != equation:
                        # Append other lines as they are
                        modified_mps_content.append(line)
                    # Append other lines as they are
                    else:
                        modified_mps_content.append(line)
            else:
                # Handle lines that don't contain valid parts (just append them as is)
                modified_mps_content.append(line)
            # if line.startswith('RHS') or line.startswith('BOUNDS'):
            #     in_columns_section = False
        
        elif after_columns_section:
            parts = line.split()
            if len(parts) >= 3:
                var_name = parts[0]
                row_name = parts[1]
                coefficient = parts[2]
                if equation == row_name:
                    print(f"Equation: {equation}")
                    modified_mps_content.append(f"    {var_name}    obj    {coefficient}\n")
                    # Replace the objective variable with the real variables
                # Append lines outside the COLUMNS section as they are
                if coefficient == objective_var:
                    continue
                else:
                    modified_mps_content.append(line)
            else:
                modified_mps_content.append(line)

        else:
            parts = line.split()
            if len(parts) >=2: 
                row_name = parts[0]
                eq_row = parts[1]
                if equation == eq_row:
                    continue
                else: 
                    modified_mps_content.append(line)
            else:
                modified_mps_content.append(line)
    for line in modified_mps_content:
        # verify the modified file has only one colums line
        columns = 0
        if 'COLUMNS' in line:
            columns += 1
        if columns == 1:
            # remove the line
            modified_mps_content.remove(line)
    

    # Save the modified MPS file
    with open(output_path, 'w') as modified_file:
        modified_file.writelines(modified_mps_content)
    print(f"Modified MPS file saved to: {output_path}")

def sections_mps_file(file):
    with open(file_path, 'r') as file:
        mps_content = file.readlines()
    for line in mps_content:
        if line.startswith('ROWS'):
            print(line)
        if line.startswith('COLUMNS'):
            print(line)
        if line.startswith('RHS'):
            print(line)
        if line.startswith('BOUNDS'):
            print(line)
        if line.startswith('ENDATA'):
            print(line)

if __name__ == '__main__':
    # Modify the MPS file in the 'data' directory
    project_root = os.path.dirname(os.getcwd())
    GAMS_path = os.path.join(project_root, 'data/real_data')
    GAMS_path_modified = os.path.join(project_root, 'data/real_data_new')
    if not os.path.exists(GAMS_path_modified):
        os.makedirs(GAMS_path_modified)
    # file_path = os.path.join(GAMS_path, 'DINAM.mps')
    # output_path = os.path.join(GAMS_path_modified, 'DINAM.mps')
    # modify_mps_objective(file_path, output_path)
    for file in os.listdir(GAMS_path):
        if file.endswith('openTEPES_EAPP_2030_sc01_st1.mps'):
            file_path = os.path.join(GAMS_path, file)
            print(os.path.exists(file_path))
            output_path = os.path.join(GAMS_path_modified, file)
            modify_mps_objective(file_path, output_path)
    