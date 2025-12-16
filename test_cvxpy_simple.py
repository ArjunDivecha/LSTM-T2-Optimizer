"""Test simple cvxpy layer."""
import torch
import cvxpy as cp
from cvxpylayers.torch import CvxpyLayer

# Simple portfolio optimization
n = 5  # 5 assets for simplicity

w = cp.Variable(n)
mu = cp.Parameter(n)
Sigma = cp.Parameter((n, n), PSD=True)  # Must be PSD (positive semidefinite)

# Constraints
constraints = [
    cp.sum(w) == 1,
    w >= 0,
]

# Try different formulations
# Formulation 1: minimize variance only
objective1 = cp.Minimize(cp.quad_form(w, Sigma))
prob1 = cp.Problem(objective1, constraints)
print(f"\nFormulation 1 (min variance): DPP = {prob1.is_dcp(dpp=True)}")

# Formulation 2: minimize -mu @ w (maximize return)
objective2 = cp.Minimize(-mu @ w)
prob2 = cp.Problem(objective2, constraints)
print(f"Formulation 2 (max return): DPP = {prob2.is_dcp(dpp=True)}")

# Formulation 3: combined
objective = cp.Minimize(-mu @ w + cp.quad_form(w, Sigma))
prob = cp.Problem(objective, constraints)

print(f"Is DCP: {prob.is_dcp()}")
print(f"Is DPP: {prob.is_dcp(dpp=True)}")

if prob.is_dcp(dpp=True):
    print("Problem is DPP-compliant!")
    layer = CvxpyLayer(prob, [mu, Sigma], [w])
    print("Layer created successfully!")
else:
    print("Problem is NOT DPP-compliant")
    print(f"Objective: {objective}")
    print(f"Constraints: {constraints}")
