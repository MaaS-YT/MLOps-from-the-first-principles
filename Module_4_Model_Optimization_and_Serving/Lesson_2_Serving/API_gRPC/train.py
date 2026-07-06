"""
gRPC ML Serving — model training side.

This script is the first step in the three-part gRPC serving pipeline:
  1. train.py   ← train a model and serialise it to disk as model.pkl
  2. server.py  ← load model.pkl, implement the gRPC Predict RPC, serve
  3. client.py  ← open a channel, call Predict, print results

The model itself is intentionally trivial (y = 2x linear regression) so that
the focus stays on the serving infrastructure, not the ML.  The same server
and proto contract would work with any sklearn, PyTorch, or ONNX model.

Run with:
    uv run train.py
"""

import numpy as np
import joblib
from sklearn.linear_model import LinearRegression


# ---------------------------------------------------------------------------
# Data
#
# Synthetic dataset for y = 2x, no noise.  The model will learn coefficient
# ~2.0 and intercept ~0.0 exactly.  The trivial data is intentional — it
# makes the server output easy to verify by inspection when testing the gRPC
# pipeline (input 3.5 → prediction 7.0, input 10.0 → prediction 20.0, etc.).
# ---------------------------------------------------------------------------

X = np.array([[1], [2], [3], [4], [5]], dtype=float)
y = np.array([2, 4, 6, 8, 10],         dtype=float)


# ---------------------------------------------------------------------------
# Model
#
# sklearn's LinearRegression uses the ordinary least squares closed-form
# solution (not gradient descent).  For this dataset it recovers the exact
# coefficients in one call to .fit().
#
# joblib.dump serialises the fitted estimator to a binary file.  The server
# loads it with joblib.load() — no re-training, no access to training data.
# joblib is preferred over pickle for numpy-heavy objects: it uses memory-
# mapped files internally, which is faster for large weight arrays.
# ---------------------------------------------------------------------------

def train_model(save_path: str = "model.pkl") -> None:
    print("Training model...")

    model = LinearRegression()
    model.fit(X, y)

    joblib.dump(model, save_path)

    print(f"Model saved to {save_path}")
    print(f"  Coefficient : {model.coef_[0]:.4f}  (expected 2.0)")
    print(f"  Intercept   : {model.intercept_:.4f}  (expected 0.0)")


if __name__ == "__main__":
    train_model()
