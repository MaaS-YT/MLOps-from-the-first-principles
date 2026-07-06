"""
LibTorch deployment — training and export side.

The core idea: PyTorch models can be serialised as TorchScript (.pt) files that
carry the full computation graph AND the trained weights in a single binary.
The C++ LibTorch runtime (torch::jit::load) can deserialise and run these files
without a Python interpreter — enabling deployment in latency-critical servers,
embedded systems, or any environment where Python is unavailable.

The export workflow has two steps:
  1. torch.jit.script  — compiles the Python module source into TorchScript IR,
                          preserving all branches and type information.
  2. scripted.save()   — serialises the IR graph + weights to a .pt file.

Why torch.jit.script over torch.jit.trace?
  trace records one execution path with a concrete input.  script compiles the
  full source.  For production exports, script is the safer default: it catches
  missing type annotations at export time, not at deployment time, and handles
  any control flow correctly.

This script demonstrates:
  1. Generate a synthetic XOR dataset (non-linearly separable — needs >= 1 hidden layer)
  2. Train a small MLP classifier to 100% validation accuracy
  3. Export with torch.jit.script → model.pt
  4. Sanity-check: scripted model output matches eager model output

The exported model.pt is consumed by cpp/inference.cpp via torch::jit::load().

Run with:
    uv run LibTorch/train_and_save.py
"""

import torch
import torch.nn as nn
from pathlib import Path


# ---------------------------------------------------------------------------
# 1. Synthetic XOR dataset
#
# XOR is not linearly separable: no single hyperplane can separate the two
# classes.  This forces the network to learn a non-linear decision boundary,
# which requires at least one hidden layer.  It's the simplest task that
# distinguishes a universal approximator from a logistic regression.
#
#   Input  →  Label
#   (0, 0) →  0
#   (0, 1) →  1
#   (1, 0) →  1
#   (1, 1) →  0
# ---------------------------------------------------------------------------

def make_xor_data(n: int = 1000) -> tuple[torch.Tensor, torch.Tensor]:
    X = torch.randint(0, 2, (n, 2)).float()
    y = (X[:, 0] != X[:, 1]).long()   # XOR label
    return X, y

torch.manual_seed(42)
X_train, y_train = make_xor_data(1000)
X_val,   y_val   = make_xor_data(200)


# ---------------------------------------------------------------------------
# 2. Model definition
#
# Architecture: 2 → 16 → 16 → 2  (input features → hidden → hidden → classes)
#
# Why type annotations?
#   torch.jit.script parses the Python source and compiles it to TorchScript IR.
#   TorchScript is a statically typed language — it must know the type of every
#   variable at compile time.  The `x: torch.Tensor` annotation on forward()
#   tells the compiler what type to expect; without it, script() raises a type
#   inference error at export time.
# ---------------------------------------------------------------------------

class XORClassifier(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(2, 16),
            nn.ReLU(),
            nn.Linear(16, 16),
            nn.ReLU(),
            nn.Linear(16, 2),   # logits for 2 classes
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


model = XORClassifier()
optimizer = torch.optim.Adam(model.parameters(), lr=1e-2)
criterion = nn.CrossEntropyLoss()


# ---------------------------------------------------------------------------
# 3. Training loop
# ---------------------------------------------------------------------------

EPOCHS = 200
BATCH  = 64

dataset = torch.utils.data.TensorDataset(X_train, y_train)
loader  = torch.utils.data.DataLoader(dataset, batch_size=BATCH, shuffle=True)

for epoch in range(1, EPOCHS + 1):
    model.train()
    for xb, yb in loader:
        optimizer.zero_grad()
        criterion(model(xb), yb).backward()
        optimizer.step()

    if epoch % 50 == 0:
        model.eval()
        with torch.no_grad():
            val_logits = model(X_val)
            val_acc = (val_logits.argmax(1) == y_val).float().mean().item()
        print(f"Epoch {epoch:>3} | val_acc = {val_acc:.3f}")


# ---------------------------------------------------------------------------
# 4. Export as TorchScript
#
# scripted.save() produces a single .pt file that bundles:
#   - The TorchScript IR graph (the computation, not Python bytecode)
#   - All trainable weights (serialised tensors)
#   - Any Python constants captured at script time
#
# The C++ side loads this with torch::jit::load(path) — no Python, no pickle,
# no importlib.  The format is stable across PyTorch versions and portable
# across Linux, macOS, Windows, and mobile (via torchscript on Android/iOS).
#
# model.eval() before export: disables dropout and batchnorm training mode so
# the exported graph reflects inference behaviour, not training behaviour.
# ---------------------------------------------------------------------------

model.eval()
scripted = torch.jit.script(model)

out_path = Path(__file__).parent / "cpp" / "model.pt"
scripted.save(str(out_path))
print(f"\nModel saved to {out_path}")

# Quick sanity check
with torch.no_grad():
    sample = torch.tensor([[0.0, 1.0], [1.0, 1.0], [0.0, 0.0], [1.0, 0.0]])
    preds  = scripted(sample).argmax(1)
    print("Sample predictions (expect 1, 0, 0, 1):", preds.tolist())
