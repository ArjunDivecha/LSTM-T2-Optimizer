"""Debug script to check data quality."""

import pickle
import numpy as np
from pathlib import Path

# Load 2010 data
year_dir = Path("data/processed/2010")

with open(year_dir / "train_X.pkl", "rb") as f:
    X_train = pickle.load(f)
with open(year_dir / "train_y.pkl", "rb") as f:
    y_train = pickle.load(f)

print("Data shapes:")
print(f"  X_train: {X_train.shape}")
print(f"  y_train: {y_train.shape}")

print("\nX_train statistics:")
print(f"  Min: {np.min(X_train)}")
print(f"  Max: {np.max(X_train)}")
print(f"  Mean: {np.mean(X_train)}")
print(f"  Std: {np.std(X_train)}")
print(f"  NaN count: {np.sum(np.isnan(X_train))}")
print(f"  Inf count: {np.sum(np.isinf(X_train))}")

print("\ny_train statistics:")
print(f"  Min: {np.min(y_train)}")
print(f"  Max: {np.max(y_train)}")
print(f"  Mean: {np.mean(y_train)}")
print(f"  Std: {np.std(y_train)}")
print(f"  NaN count: {np.sum(np.isnan(y_train))}")
print(f"  Inf count: {np.sum(np.isinf(y_train))}")

# Check for any weird values
if np.sum(np.isnan(X_train)) > 0:
    print("\n❌ FAIL: NaN values in X_train!")
if np.sum(np.isinf(X_train)) > 0:
    print("\n❌ FAIL: Inf values in X_train!")
if np.sum(np.isnan(y_train)) > 0:
    print("\n❌ FAIL: NaN values in y_train!")
if np.sum(np.isinf(y_train)) > 0:
    print("\n❌ FAIL: Inf values in y_train!")
