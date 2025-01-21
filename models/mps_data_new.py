import os

#from mps_data import detect_objective_and_coefficients_new

def parse_mps(file_path):
    """
    Parse the MPS file into structured data.

    Returns a dictionary with keys:
    - 'NAME': problem name (if found)
    - 'ROWS': a list of tuples (row_type, row_name)
    - 'COLUMNS': a dict of {var_name: {row_name: coefficient}}
    - 'RHS': a dict of {rhs_name: {row_name: value}}
    - 'BOUNDS': a dict of {bound_name: {var_name: (bound_type, value)}}
    - 'RANGES': a dict of {range_name: {row_name: range_value}}
    - 'ENDATA': boolean indicating end of data

    Only 'COLUMNS' are strictly needed for the objective detection, but we parse more
    for completeness and potential future use.
    """

    sections = ['NAME', 'ROWS', 'COLUMNS', 'RHS', 'RANGES', 'BOUNDS', 'ENDATA']
    current_section = None

    data = {
        'NAME': None,
        'ROWS': [],
        'COLUMNS': {},
        'RHS': {},
        'RANGES': {},
        'BOUNDS': {},
        'ENDATA': False
    }

    with open(file_path, 'r') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            # Check if line indicates a new section
            upper_line = line.upper()
            if upper_line in sections:
                current_section = upper_line
                if current_section == 'ENDATA':
                    data['ENDATA'] = True
                continue

            # Parse line depending on the current section
            if current_section == 'NAME':
                # The NAME section typically has one line with problem name
                if data['NAME'] is None:
                    data['NAME'] = line
                # If multiple lines appear under NAME (rare), take the first as problem name.
            
            elif current_section == 'ROWS':
                # Format: row_type row_name (e.g. N  OBJROW)
                parts = line.split()
                if len(parts) == 2:
                    row_type, row_name = parts
                    data['ROWS'].append((row_type, row_name))
            
            elif current_section == 'COLUMNS':
                # Format can be:
                #   VAR_NAME   ROW_NAME   VALUE   [ROW_NAME   VALUE]
                # Each line can define 1 or 2 entries.
                parts = line.split()
                # Expecting at least 3 parts: var_name, row_name, value
                # Possibly 5 parts: var_name, row_name1, value1, row_name2, value2
                if len(parts) < 3:
                    continue
                var_name = parts[0]
                # Make sure var_name is in dictionary
                if var_name not in data['COLUMNS']:
                    data['COLUMNS'][var_name] = {}

                # Process first triple
                row_name = parts[1]
                value = float(parts[2])
                data['COLUMNS'][var_name][row_name] = value
                # If there's a second triple on the same line
                if len(parts) == 5:
                    row_name2 = parts[3]
                    value2 = float(parts[4])
                    data['COLUMNS'][var_name][row_name2] = value2

            elif current_section == 'RHS':
                # Format similar to COLUMNS: RHS_NAME ROW_NAME VALUE [ROW_NAME VALUE]
                parts = line.split()
                if len(parts) < 3:
                    continue
                rhs_name = parts[0]
                if rhs_name not in data['RHS']:
                    data['RHS'][rhs_name] = {}
     
                row_name = parts[1]
                value = float(parts[2])
                data['RHS'][rhs_name][row_name] = value
                if len(parts) == 5:
                    row_name2 = parts[3]
                    value2 = float(parts[4])
                    data['RHS'][rhs_name][row_name2] = value2

            elif current_section == 'RANGES':
                # Format: RANGE_NAME ROW_NAME VALUE [ROW_NAME VALUE]
                parts = line.split()
                if len(parts) < 3:
                    continue
                range_name = parts[0]
                if range_name not in data['RANGES']:
                    data['RANGES'][range_name] = {}

                row_name = parts[1]
                value = float(parts[2])
                data['RANGES'][range_name][row_name] = value

                if len(parts) == 5:
                    row_name2 = parts[3]
                    value2 = float(parts[4])
                    data['RANGES'][range_name][row_name2] = value2

            elif current_section == 'BOUNDS':
                # Format: BOUND_TYPE BOUND_NAME VAR_NAME VALUE?
                # e.g. LO BND1 X1 0
                parts = line.split()
                if len(parts) >= 3:
                    bound_type = parts[0]
                    bound_name = parts[1]
                    var_name = parts[2]
                    val = None
                    if len(parts) == 4:
                        val = float(parts[3])
                    if bound_name not in data['BOUNDS']:
                        data['BOUNDS'][bound_name] = {}
                    data['BOUNDS'][bound_name][var_name] = (bound_type, val)

    return data


def detect_objective_and_coefficients_new(file_path):
    """
    Detect the auxiliary objective variable and the coefficients of the actual variables linked to it.
    The function handles multiple unconstrained (N) rows as follows:
    - If an 'obj' row is found, it uses that as the objective without warning.
    - If no 'obj' row is found but exactly one N row exists, it uses that one silently.
    - If no 'obj' row is found and more than one N row exists, it issues a warning and uses the first one.

    Parameters:
    - file_path: Path to the MPS file.

    Returns:
    - objective_var: The detected auxiliary variable used in the objective (e.g., 'x7').
    - variable_coeffs: A dictionary where keys are variable names and values are their coefficients.
    - equation: The row name (constraint) representing the actual objective function.
    """
    data = parse_mps(file_path)

    # Identify all unconstrained (N) rows
    n_rows = [row_name for (row_type, row_name) in data['ROWS'] if row_type.upper() == 'N']
    # Attempt to find a row named 'obj' among the N rows
    objective_row_name = None
    for rname in n_rows:
        if rname.lower() == 'obj':
            objective_row_name = rname
            break

    # If 'obj' not found:
    if objective_row_name is None:
        if len(n_rows) > 1:
            # Multiple N rows, no 'obj'
            print("Warning: more than one unconstrained row, using the first one as the objective.")
            objective_row_name = n_rows[0]
        elif len(n_rows) == 1:
            # Single N row, no 'obj', use it silently
            objective_row_name = n_rows[0]
   
    print(f"Objective row name: {objective_row_name}")
    # If still no objective row found, return early
    if objective_row_name is None:
        return None, {}, None

    # Find candidate objective variables (variables that appear in the objective row)
    candidate_vars = [var for var, row_dict in data['COLUMNS'].items() if objective_row_name in row_dict]
    # Identify the auxiliary objective variable as one that links the objective row to exactly one other row
    objective_var = None
    equation = None
    for var in candidate_vars:
        other_rows = [r for r in data['COLUMNS'][var].keys() if r != objective_row_name]
        print(f"Variable: {var}, other rows: {other_rows}")
        if len(other_rows) == 1:
            objective_var = var
            equation = other_rows[0]
            break

    if not objective_var or not equation:
        print("Warning: could not identify a unique auxiliary variable or equation.")
        # Could not identify a unique auxiliary variable or equation
        return None, {}, None

    # Gather coefficients of all variables in the equation row
    variable_coeffs = {var: row_dict[equation] for var, row_dict in data['COLUMNS'].items() if equation in row_dict}
    print(f"Detected objective variable: {objective_var}")
    print(f"Equation row: {equation}")
    return objective_var, variable_coeffs, equation



def modify_mps_objective(file_path, output_path):
    """
    Modify an MPS file to replace an auxiliary variable in the objective function with
    a linear combination of other variables, and change the sign of the coefficients depending 
    on whether we are minimizing or maximizing.

    Steps:
    1. Parse the MPS file.
    2. Detect the auxiliary objective variable, the coefficients of the actual variables linked to it, and the equation row.
    3. Determine if we are minimizing or maximizing based on the sign of the auxiliary variable 
       in the objective row (heuristic).
    4. Remove the auxiliary variable from the objective row.
    5. Add the corresponding coefficients from the equation into the objective row, with appropriate sign changes.
    6. Remove the equation row and its references from the MPS data.

    Parameters:
    - file_path: Path to the input MPS file.
    - output_path: Path to the output modified MPS file.
    """
    data = parse_mps(file_path)
    
    # Detect the auxiliary objective variable, variable coefficients, and equation row
    objective_var, variable_coeffs, equation = detect_objective_and_coefficients_new(file_path)
    print(objective_var, variable_coeffs, equation)
    if not objective_var or not variable_coeffs or not equation:
        raise ValueError("No suitable auxiliary objective variable or coefficients detected.")

    # Remove the objective_var from variable_coeffs to avoid double counting
    if objective_var in variable_coeffs:
        del variable_coeffs[objective_var]

    # Determine direction (minimize or maximize) based on the sign of the auxiliary variable's obj coefficient
    obj_coeff = data['COLUMNS'].get(objective_var, {}).get('obj', 0.0)
    if obj_coeff > 0:
        direction = 'min'
    elif obj_coeff < 0:
        direction = 'max'


    # Remove the auxiliary variable from the objective row
    if objective_var in data['COLUMNS']:
        if 'OBJ' in data['COLUMNS'][objective_var]:
            del data['COLUMNS'][objective_var]['OBJ']
            if not data['COLUMNS'][objective_var]:
                del data['COLUMNS'][objective_var]

    # Add substituted variables (from equation) into the objective row
    for var, coeff in variable_coeffs.items():
        if var not in data['COLUMNS']:
            data['COLUMNS'][var] = {}
        new_coeff = -coeff if direction == 'min' else coeff
        data['COLUMNS'][var]['OBJ'] = new_coeff

    # Remove the equation row from all variables
    for var in list(data['COLUMNS'].keys()):
        if equation in data['COLUMNS'][var]:
            del data['COLUMNS'][var][equation]
            if not data['COLUMNS'][var]:
                del data['COLUMNS'][var]

    # If the equation row exists in the RHS section, rename it to 'OBJ'
    if equation in data['RHS']:
        data['RHS']['OBJ'] = data['RHS'][equation]
        del data['RHS'][equation]
    
    if equation in data['RANGES']:
        data['RANGES']['OBJ'] = data['RANGES'][equation]
        del data['RANGES'][equation]
    
    if objective_var in data['BOUNDS']:
        data['BOUNDS']['OBJ'] = data['BOUNDS'][objective_var]
        del data['BOUNDS'][equation]

    # Remove the equation row from ROWS section
    data['ROWS'] = [(row_type, row_name) for row_type, row_name in data['ROWS'] if row_name != equation]

    # Reconstruct and write the modified MPS file
    lines = []
    if data['NAME']:
        lines.append(f"NAME          {data['NAME']}\n")
    else:
        lines.append("NAME          PROBLEM\n")

    # ROWS section
    lines.append("ROWS\n")
    found_obj_row = any(row_name == 'OBJ' for _, row_name in data['ROWS'])
    if not found_obj_row:
        lines.append(" N  OBJ\n")
    for row_type, row_name in data['ROWS']:
        lines.append(f" {row_type}  {row_name}\n")

    # COLUMNS section
    lines.append("COLUMNS\n")
    for var in sorted(data['COLUMNS'].keys()):
        for row_name, value in data['COLUMNS'][var].items():
            lines.append(f"  {var}  {row_name}  {value:g}\n")

    # RHS section
    if data['RHS']:
        lines.append("RHS\n")
        for rhs_name, rhs_entries in data['RHS'].items():
            for row_name, value in rhs_entries.items():
                lines.append(f"  {rhs_name}  {row_name}  {value:g}\n")

    # RANGES section
    if data['RANGES']:
        lines.append("RANGES\n")
        for range_name, ranges_entries in data['RANGES'].items():
            for row_name, value in ranges_entries.items():
                lines.append(f"  {range_name}  {row_name}  {value:g}\n")

    # BOUNDS section
    if data['BOUNDS']:
        lines.append("BOUNDS\n")
        for bound_name, bounds_entries in data['BOUNDS'].items():
            for var_name, (bound_type, value) in bounds_entries.items():
                if value is not None:
                    lines.append(f" {bound_type} {bound_name} {var_name} {value:g}\n")
                else:
                    lines.append(f" {bound_type} {bound_name} {var_name}\n")

    # ENDATA section
    lines.append("ENDATA\n")

    # Write the modified MPS file
    with open(output_path, 'w') as f:
        f.writelines(lines)

    print(f"Modified MPS file saved to: {output_path}")


if __name__ == '__main__':
    # Modify the MPS file in the 'data' directory
    project_root = os.path.dirname(os.getcwd())
    GAMS_path = os.path.join(project_root, 'data/GAMS_library')
    GAMS_path_modified = os.path.join(project_root, 'data/real_data_new')
    if not os.path.exists(GAMS_path_modified):
        os.makedirs(GAMS_path_modified)
    # file_path = os.path.join(GAMS_path, 'DINAM.mps')
    # output_path = os.path.join(GAMS_path_modified, 'DINAM.mps')
    # modify_mps_objective(file_path, output_path)
    for file in os.listdir(GAMS_path):
        if  file.endswith('.mps') and not file.endswith('ORANI.mps'):
            file_path = os.path.join(GAMS_path, file)
            print(file)
            output_path = os.path.join(GAMS_path_modified, file)
            modify_mps_objective(file_path, output_path)
            # objective_var, variable_coeffs, equation = detect_objective_and_coefficients_new(file_path)
            # objective_var1, variable_coeffs1, equation1 = detect_objective_and_coefficients_new1(file_path)
            # print(objective_var, variable_coeffs, equation)
            # print(objective_var1, variable_coeffs1, equation1)