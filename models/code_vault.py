import numpy as np
import gurobipy as gp
from gurobipy import GRB

from utils_models.utils_functions import *


def standard_form1(originañ_model):

    model = originañ_model.copy()
    # Step 1: Ensure the model is in minimization form
    if model.ModelSense != 1:
        model.setObjective(-1 * model.getObjective(), GRB.MINIMIZE)
        model.ModelSense = 1  # Set the model sense to minimization

    model.update()
    # Step 2: Ensure all variables are non-negative
    for var in model.getVars():
        if var.LB != 0 or var.UB < GRB.INFINITY or var.LB == -GRB.INFINITY:
            # Create a new variable with lb=0 and no upper bound
            var_new = model.addVar(lb=-GRB.INFINITY, vtype=var.VType, name=f"{var.VarName}_new")
            model.update()

            # Replace 'var' in constraints with 'var_new' and enforce the original bounds
            for constr in model.getConstrs():
                coeff = model.getCoeff(constr, var)
                if coeff != 0:
                    model.chgCoeff(constr, var_new, coeff)
                    #model.chgCoeff(constr, var, 0)

            # Add constraints to enforce original bounds if needed
            if var.LB > -GRB.INFINITY and var.LB != 0:
                    model.addConstr(var_new >= var.LB, name=f"{var.VarName}_LB")
            if var.UB < GRB.INFINITY:
                    model.addConstr(var_new <= var.UB, name=f"{var.VarName}_UB")

            # Replace 'var' in the objective function
            obj_coeff = var.getAttr(GRB.Attr.Obj)
            if obj_coeff != 0:
                model.setObjective(model.getObjective() + obj_coeff * var_new, GRB.MINIMIZE)

            # Remove the original variable from the model
            model.remove(var)

    model.update()

    # Step 3: Transform all constraints into equalities by introducing slack/surplus variables
    for constr in model.getConstrs():
        sense = constr.Sense
        if constr.ConstrName == 'obj':
            continue  # Skip any handling for the objective constraint

        if sense != GRB.EQUAL:
            # Add slack or surplus variable based on constraint type
            slack_var = model.addVar(lb=0, name=f"slack_{constr.ConstrName}")
            model.update()

            # Add slack or surplus depending on the sense of the constraint
            if sense == GRB.LESS_EQUAL:
                model.chgCoeff(constr, slack_var, 1)
            if sense == GRB.GREATER_EQUAL:
                model.chgCoeff(constr, slack_var, -1)
            # Set the constraint to equality after adding slack/surplus
            constr.Sense = GRB.EQUAL

    model.update()

    return model

def standard_form2(model): 
    # Step 1: Ensure the model is in minimization form 
    if model.ModelSense != 1: 
        model.setObjective(-1 * model.getObjective(), GRB.MINIMIZE) 
        model.ModelSense = 1  # Set the model sense to minimization 
 
    # Step 2: Ensure all variables are non-negative 
    for var in model.getVars():
        if var.LB < 0: 
            # Replace variable with two non-negative variables
            pos_var = model.addVar(lb=0, vtype=var.VType, name=f"{var.VarName}_pos")
            neg_var = model.addVar(lb=0, vtype=var.VType, name=f"{var.VarName}_neg")
            # Update constraints to replace old variable with the new positive and negative parts
            for constr in model.getConstrs():
                coeff = model.getCoeff(constr, var)
                if coeff != 0:
                    model.chgCoeff(constr, pos_var, coeff)
                    model.chgCoeff(constr, neg_var, -coeff)
                    model.chgCoeff(constr, var, 0)

            # Update the objective function if the original variable was part of it
            obj_coeff = var.getAttr(GRB.Attr.Obj)
            if obj_coeff != 0:
                model.setObjective(model.getObjective() + obj_coeff * (pos_var - neg_var), GRB.MINIMIZE)

            # Remove the original variable from the model
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
 
    return model 