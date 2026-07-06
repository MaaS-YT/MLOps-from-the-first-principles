"""
Generate reference data for drift monitoring
"""
import numpy as np
import pandas as pd

np.random.seed(42)

# Reference (training) data
ref = pd.DataFrame({
    "age": np.random.normal(30, 4, 1000),
    "income": np.random.normal(60000, 10000, 1000),
    "transactions": np.random.normal(12, 3, 1000)
})

# Save reference data
ref.to_csv("reference.csv", index=False)
print(f"Generated reference.csv with {len(ref)} records")
print(f"\nReference data statistics:")
print(ref.describe())
