"""
Quantization — from first principles.

Standard neural networks store weights and activations as FP32 (32-bit float).
Quantization maps these to lower-precision integer types (usually INT8).

Why this works:
  - Weights in a trained network cluster in narrow ranges. The information
    content of a weight is its *relative position* in that range, not its
    absolute FP32 value. INT8 (256 levels) is sufficient for most tasks.
  - Hardware integer SIMD units (VNNI on Intel, SDOT on ARM) run 2–4× faster
    than their FP32 equivalents and use narrower memory buses.

This script demonstrates three quantization modes on the same MLP baseline:
  1. FP32 baseline (reference)
  2. Dynamic quantization — weights to INT8 at conversion time
  3. Static PTQ (Post-Training Quantization) — weights + activations to INT8

Run with:
    uv run --no-sync python qnt.py

Note: PyTorch quantization runs on CPU only.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import random
import numpy as np
import os
import time


# ---------------------------------------------------------------------------
# Reproducibility and device setup
# ---------------------------------------------------------------------------

SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

# PyTorch quantization only runs on CPU.
# Training is done on GPU if available, then the model is moved to CPU
# before quantization.
TRAIN_DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Training device: {TRAIN_DEVICE}")


# ---------------------------------------------------------------------------
# Data
# ---------------------------------------------------------------------------

transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.1307,), (0.3081,))
])

train_ds = datasets.MNIST(root="./data", train=True,  download=True, transform=transform)
test_ds  = datasets.MNIST(root="./data", train=False, download=True, transform=transform)

train_loader = DataLoader(train_ds, batch_size=128, shuffle=True,  num_workers=2, pin_memory=True)
test_loader  = DataLoader(test_ds,  batch_size=256, shuffle=False, num_workers=2, pin_memory=True)

# calib_loader is used during the PTQ calibration pass.
# A small number of batches (640 samples here) is enough to observe the
# activation range. More data → more accurate scales, diminishing returns past ~1000.
calib_loader = DataLoader(train_ds, batch_size=64, shuffle=True)


# ---------------------------------------------------------------------------
# Model
#
# Simple MLP: 784 → 256 → 128 → 10.
# We define two versions:
#   MLP              — standard, for training and dynamic quantization
#   QuantizableMLP   — same weights, with QuantStub/DeQuantStub wrappers for PTQ
# ---------------------------------------------------------------------------

class MLP(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(28 * 28, 256)
        self.fc2 = nn.Linear(256, 128)
        self.fc3 = nn.Linear(128, 10)

    def forward(self, x):
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return self.fc3(x)


class QuantizableMLP(nn.Module):
    """
    Same architecture as MLP, with quantization boundary markers.

    QuantStub marks the point where the computation transitions from
    FP32 (the input) to INT8 (inside the model). It inserts a quantize_per_tensor
    op in the quantized graph: scale + zero_point are determined during calibration.

    DeQuantStub marks where INT8 computation returns to FP32. This is typically
    just before the loss function, which expects FP32.

    Without these stubs, PyTorch doesn't know where the quantized region starts
    and ends. The calibration pass instruments the stubs to observe input ranges.

    Also note: nn.ReLU is used as a module (not F.relu) so that quantization
    can fuse Conv/Linear → ReLU → Quantize into a single INT8 kernel.
    F.relu is a function call — the graph transformer can't fuse it.
    """
    def __init__(self):
        super().__init__()
        self.quant   = torch.quantization.QuantStub()
        self.fc1     = nn.Linear(28 * 28, 256)
        self.fc2     = nn.Linear(256, 128)
        self.fc3     = nn.Linear(128, 10)
        self.relu    = nn.ReLU()
        self.dequant = torch.quantization.DeQuantStub()

    def forward(self, x):
        x = x.view(x.size(0), -1)
        x = self.quant(x)                # FP32 → INT8
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        x = self.fc3(x)
        return self.dequant(x)           # INT8 → FP32


# ---------------------------------------------------------------------------
# Training and evaluation helpers
# ---------------------------------------------------------------------------

def train_one_epoch(model, loader, opt, device):
    model.train()
    total_loss = correct = total = 0
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        opt.zero_grad()
        logits = model(x)
        loss   = F.cross_entropy(logits, y)
        loss.backward()
        opt.step()
        total_loss += loss.item() * x.size(0)
        correct    += (logits.argmax(1) == y).sum().item()
        total      += x.size(0)
    return correct / total, total_loss / total


@torch.no_grad()
def evaluate(model, loader, device="cpu"):
    model.eval()
    correct = total = 0
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        correct += (model(x).argmax(1) == y).sum().item()
        total   += y.size(0)
    return 100.0 * correct / total


def file_kb(path: str) -> float:
    return os.path.getsize(path) / 1024 if os.path.exists(path) else 0.0


def latency_ms(model, loader, device="cpu", n_batches=20):
    model.eval()
    t0 = time.perf_counter()
    with torch.no_grad():
        for i, (x, _) in enumerate(loader):
            model(x.to(device))
            if i >= n_batches:
                break
    return (time.perf_counter() - t0) / (n_batches + 1) * 1000


# ---------------------------------------------------------------------------
# SECTION 1 — FP32 baseline
# ---------------------------------------------------------------------------

print("=" * 60)
print("SECTION 1 — FP32 baseline")
print("=" * 60)

model = MLP().to(TRAIN_DEVICE)
opt   = torch.optim.Adam(model.parameters(), lr=1e-3)

for ep in range(3):
    tr_acc, tr_loss = train_one_epoch(model, train_loader, opt, TRAIN_DEVICE)
    te_acc          = evaluate(model, test_loader, TRAIN_DEVICE)
    print(f"  epoch {ep+1}/3 | loss {tr_loss:.4f} | train {tr_acc:.4f} | test {te_acc:.2f}%")

torch.save(model.state_dict(), "fp32.pth")
fp32_acc  = evaluate(model, test_loader, TRAIN_DEVICE)
fp32_size = file_kb("fp32.pth")
print(f"\nFP32 — size: {fp32_size:.1f} KB | acc: {fp32_acc:.2f}%")


# ---------------------------------------------------------------------------
# SECTION 2 — Dynamic quantization
#
# How it works:
#   `quantize_dynamic` walks the module tree and replaces each nn.Linear (and
#   optionally nn.LSTM, nn.GRU) with a DynamicQuantizedLinear module.
#
#   Weight quantization happens once at conversion time:
#     w_int8, scale, zero_point = quantize_per_channel(w_fp32)
#   The INT8 weights are stored in the new module.
#
#   Activation quantization happens at every forward pass:
#     x_int8 = quantize_per_tensor(x_fp32, scale=dynamic_scale)
#   where dynamic_scale is chosen fresh for each input tensor by examining
#   its min/max range. This is the "dynamic" in dynamic quantization.
#
#   Computation: INT8 × INT8 matmul, result accumulated in INT32, then cast
#   back to FP32 before output. The accumulate-in-INT32 step is critical —
#   INT8 × INT8 sums can overflow if accumulated in INT8 directly.
#
# Tradeoffs:
#   + Zero calibration data needed
#   + Works on any model instantly
#   - Activations require per-batch scale computation (small overhead)
#   - Activation quantization is less precise than static (no calibration)
#   - Memory bandwidth savings are partial (weights INT8, activations FP32 until runtime)
# ---------------------------------------------------------------------------

print("\n" + "=" * 60)
print("SECTION 2 — Dynamic quantization")
print("=" * 60)

# Move to CPU before quantization — PyTorch quantized operators are CPU-only
model_cpu = MLP()
model_cpu.load_state_dict(
    torch.load("fp32.pth", map_location="cpu", weights_only=True)
)
model_cpu.eval()

model_dynamic = torch.quantization.quantize_dynamic(
    model_cpu,
    {nn.Linear},           # which module types to quantize
    dtype=torch.qint8      # target dtype for weights
)

# Inspect what happened to fc1:
print(f"\nOriginal fc1 type: {type(model_cpu.fc1)}")
print(f"Quantized fc1 type: {type(model_dynamic.fc1)}")
# The weight is now a PackedLinearWeight — INT8 data with scale/zero_point metadata

torch.save(model_dynamic.state_dict(), "dynamic_int8.pth")
dyn_acc  = evaluate(model_dynamic, test_loader, "cpu")
dyn_size = file_kb("dynamic_int8.pth")
print(f"\nDynamic INT8 — size: {dyn_size:.1f} KB | acc: {dyn_acc:.2f}%")
print(f"Size vs FP32: {fp32_size / dyn_size:.2f}×")


# ---------------------------------------------------------------------------
# SECTION 3 — Static Post-Training Quantization (PTQ)
#
# How it works:
#   Static PTQ quantizes both weights AND activations to INT8. Because
#   activation ranges vary by input, we need to observe them on representative
#   data before converting — this is the calibration pass.
#
#   Step 1 — Prepare:
#     `torch.quantization.prepare()` inserts Observer modules after every
#     activation point (after QuantStub, after each ReLU, etc.). Observers
#     collect min/max statistics during calibration.
#
#   Step 2 — Calibrate:
#     Run a few hundred forward passes with real data. No labels needed;
#     no gradients computed. The observers record activation statistics.
#
#   Step 3 — Convert:
#     `torch.quantization.convert()` replaces:
#       - nn.Linear → QuantizedLinear (weights + input scale pre-determined)
#       - QuantStub → quantize_per_tensor (uses calibrated scale)
#       - Observers are removed
#     The result is a fully INT8 graph with no runtime scale computation.
#
#   qconfig = "fbgemm" selects:
#     - HistogramObserver for weights (precise, uses percentile clamping)
#     - MovingAverageMinMaxObserver for activations (stable across batches)
#     - fbgemm backend (Intel x86 quantized matmul library)
#   Use "qnnpack" on ARM (mobile/Raspberry Pi).
#
# Tradeoffs:
#   + Weights AND activations are INT8 → maximum memory + compute savings
#   + No runtime scale computation → pure INT8 kernels throughout
#   - Requires calibration data
#   - Requires QuantStub/DeQuantStub wrappers in the model
#   - Slightly more accurate than dynamic (pre-computed scales fit the data distribution)
# ---------------------------------------------------------------------------

print("\n" + "=" * 60)
print("SECTION 3 — Static PTQ")
print("=" * 60)

model_static = QuantizableMLP()
model_static.load_state_dict(
    torch.load("fp32.pth", map_location="cpu", weights_only=True),
    strict=False   # QuantStub and DeQuantStub have no weights in fp32.pth
)
model_static.eval()

# Assign qconfig: which observer and quantization scheme to use
model_static.qconfig = torch.quantization.get_default_qconfig("fbgemm")

# Step 1: Insert observers at every activation point
torch.quantization.prepare(model_static, inplace=True)
print(f"\nAfter prepare — fc1 type: {type(model_static.fc1)}")
# Note: fc1 is still nn.Linear here — prepare only adds observers, doesn't replace

# Step 2: Calibration pass — ~640 samples is enough for a toy model
print("Running calibration pass...")
with torch.no_grad():
    for i, (x, _) in enumerate(calib_loader):
        model_static(x)
        if i >= 10:    # 10 batches × 64 samples = 640 total
            break

# After calibration, observers have collected activation statistics.
# Examine what was recorded:
print(f"Quant stub scale after calibration: {model_static.quant.activation_post_process.calculate_qparams()}")

# Step 3: Convert to INT8 — replaces Linear, fuses ops, removes observers
torch.quantization.convert(model_static, inplace=True)
print(f"\nAfter convert — fc1 type: {type(model_static.fc1)}")
# Now it's nnq.Linear (quantized linear) — INT8 weights + INT8 activations

torch.save(model_static.state_dict(), "int8.pth")
static_acc  = evaluate(model_static, test_loader, "cpu")
static_size = file_kb("int8.pth")
print(f"\nStatic INT8 — size: {static_size:.1f} KB | acc: {static_acc:.2f}%")
print(f"Size vs FP32: {fp32_size / static_size:.2f}×")


# ---------------------------------------------------------------------------
# SECTION 4 — Comparison
# ---------------------------------------------------------------------------

print("\n" + "=" * 60)
print("SECTION 4 — Comparison")
print("=" * 60)

# Move FP32 model to CPU for a fair latency comparison
model.to("cpu")

fp32_lat   = latency_ms(model,         test_loader)
dyn_lat    = latency_ms(model_dynamic,  test_loader)
static_lat = latency_ms(model_static,   test_loader)

print(f"\n{'':26} {'Size KB':>8} {'Acc %':>8} {'ms/batch':>10}")
print("-" * 56)
print(f"{'FP32 baseline':26} {fp32_size:>8.1f} {fp32_acc:>8.2f} {fp32_lat:>10.1f}")
print(f"{'Dynamic INT8':26} {dyn_size:>8.1f} {dyn_acc:>8.2f} {dyn_lat:>10.1f}")
print(f"{'Static INT8 (PTQ)':26} {static_size:>8.1f} {static_acc:>8.2f} {static_lat:>10.1f}")

print("""
Key observations:
  - Size reduction:  weights are packed INT8, roughly 4× smaller on disk
  - Accuracy:        typically < 0.5 pp drop on simple models; larger models
                     may need QAT (Quantization-Aware Training) for < 1 pp
  - Latency:         depends on CPU; VNNI (Intel Cascade Lake+) and SDOT (ARMv8.2+)
                     give the full 4× speedup; older CPUs may show less benefit
  - Dynamic vs PTQ:  PTQ is generally faster and slightly more accurate because
                     activation scales are pre-computed rather than determined
                     fresh for every input batch
""")

# Cleanup
for f in ("fp32.pth", "dynamic_int8.pth", "int8.pth"):
    if os.path.exists(f):
        os.remove(f)
