import pulp
import os 
import gurobipy as gp

real_data = os.path.join(os.getcwd(), "real_data")
problem1 = os.path.join(real_data, "openTEPES_nt2030_2030_CY2009_st1.lp")
problem2 = os.path.join(real_data, "openTEPES_NT2040_2040_CY2009_st0.lp")
output_file1 = os.path.join(real_data, "openTEPES_nt2030_2030_CY2009_st1.mps")
output_file2 = os.path.join(real_data, "openTEPES_NT2040_2040_CY2009_st0.mps")

# Load the LP problem from a file
with open(problem1, "r") as file:
    lp_problem = pulp.LpProblem.from_lp_string(file.read())
# Write the problem to an MPS file
lp_problem.writeMPS(output_file1)

# Load the LP problem from a file
# lp_problem = pulp.LpProblem()
# lp_problem.readLP(problem2)


print("Conversion completed! The MPS file has been saved as 'output_file.mps'")
