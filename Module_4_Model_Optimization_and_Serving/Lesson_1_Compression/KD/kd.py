"""
Knowledge Distillation — from first principles.

The core idea: a large pre-trained model (teacher) has learned a rich internal
representation of the data. Its output probability distribution over all classes
carries information that hard labels (0 or 1) discard. A small model (student)
trained to mimic this distribution learns more than one trained on ground truth alone.

This script demonstrates:
  1. Train a CNN teacher on MNIST
  2. Train a student MLP from scratch (baseline — no distillation)
  3. Train the same student MLP with knowledge distillation
  4. Compare: params, accuracy, inference speed

Run with:
    uv run --no-sync python kd.py
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import datasets, transforms
import random
import numpy as np
import time
from torch.utils.data import DataLoader


# ---------------------------------------------------------------------------
# Reproducibility
# ---------------------------------------------------------------------------

SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.backends.cudnn.benchmark = True   # cuDNN auto-tuner: finds fastest conv algorithm

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
USE_AMP = torch.cuda.is_available()    # Automatic Mixed Precision: FP16 compute, FP32 accumulation

print(f"Device: {DEVICE}")
if DEVICE.type == "cuda":
    print(f"CUDA:   {torch.version.cuda} | GPU: {torch.cuda.get_device_name(0)}")


# ---------------------------------------------------------------------------
# Data
# ---------------------------------------------------------------------------

# MNIST normalization constants (mean=0.1307, std=0.3081) computed over the
# training set. Normalizing makes the input distribution roughly N(0,1),
# which keeps gradients well-scaled at initialization.
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.1307,), (0.3081,))
])

train_ds = datasets.MNIST(root="data", train=True,  download=True, transform=transform)
test_ds  = datasets.MNIST(root="data", train=False, download=True, transform=transform)

# pin_memory=True keeps the batch tensor in pinned (page-locked) host memory,
# so the GPU DMA engine can transfer it without CPU involvement — faster H→D copy.
train_loader = DataLoader(train_ds, batch_size=128, shuffle=True,  num_workers=2, pin_memory=DEVICE.type == "cuda")
test_loader  = DataLoader(test_ds,  batch_size=256, shuffle=False, num_workers=2, pin_memory=DEVICE.type == "cuda")


# ---------------------------------------------------------------------------
# Models
#
# Teacher: CNN — 1.2M parameters, learns spatial hierarchies via convolutions
# Student: MLP — 16K parameters, 75× smaller, no spatial inductive bias
#
# Without distillation the student underfits — too few parameters to learn
# the raw label signal from scratch. Distillation bridges the gap by giving
# the student a richer training signal: the teacher's soft distribution.
# ---------------------------------------------------------------------------

class TeacherNet(nn.Module):
    """
    Two conv layers followed by two FC layers.

    Conv architecture:
      input:  [B, 1, 28, 28]
      conv1:  [B, 32, 26, 26]   (3×3 kernel, no padding → 28-2=26)
      conv2:  [B, 64, 24, 24]   (3×3 kernel again → 26-2=24)
      pool:   [B, 64, 12, 12]   (MaxPool2d(2) halves spatial dims)
      flatten:[B, 9216]          (64 * 12 * 12 = 9216)
      fc1:    [B, 128]
      fc2:    [B, 10]            (logits — no softmax here, CE applies it)
    """
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 32, 3, 1)
        self.conv2 = nn.Conv2d(32, 64, 3, 1)
        self.fc1   = nn.Linear(9216, 128)
        self.fc2   = nn.Linear(128, 10)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.max_pool2d(x, 2)
        x = torch.flatten(x, 1)
        x = F.relu(self.fc1(x))
        return self.fc2(x)


class StudentNet(nn.Module):
    """
    Minimal MLP: flatten → FC(20) → FC(10).

    Fewer parameters mean fewer degrees of freedom to fit the training set.
    Hard-label training gives it 60,000 binary signals: "this is a 3" (or not).
    Distillation gives it 60,000 × 10-dimensional signals: the teacher's belief
    distribution over all digits, e.g. [0.01, 0.01, 0.05, 0.87, 0.02, ...].
    The inter-class structure (3 is often confused with 8) is preserved.
    """
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(28 * 28, 20)
        self.fc2 = nn.Linear(20, 10)

    def forward(self, x):
        x = torch.flatten(x, 1)
        x = F.relu(self.fc1(x))
        return self.fc2(x)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

@torch.no_grad()
def accuracy(model: nn.Module, loader: DataLoader) -> float:
    model.eval()
    correct = total = 0
    for x, y in loader:
        x, y = x.to(DEVICE), y.to(DEVICE)
        pred = model(x).argmax(1)
        correct += (pred == y).sum().item()
        total   += y.size(0)
    return 100.0 * correct / max(1, total)


def count_params(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters())


def latency_ms(model: nn.Module, loader: DataLoader, n_batches: int = 30) -> float:
    """Average batch inference time in milliseconds."""
    model.eval()
    t0 = time.perf_counter()
    with torch.no_grad():
        for i, (x, _) in enumerate(loader):
            model(x.to(DEVICE))
            if i >= n_batches:
                break
    return (time.perf_counter() - t0) / (n_batches + 1) * 1000


# ---------------------------------------------------------------------------
# SECTION 1 — Train the teacher
#
# Standard cross-entropy training. AMP (autocast + GradScaler) halves GPU
# memory usage by computing forward/backward in FP16 while keeping master
# weights in FP32 (which is necessary for gradient update precision).
#
# GradScaler prevents FP16 underflow: it scales the loss up before backward,
# then unscales before the optimizer step, and skips the step entirely if any
# gradient is NaN or Inf (which can happen transiently in FP16).
# ---------------------------------------------------------------------------

print("\n" + "=" * 60)
print("SECTION 1 — Train teacher")
print("=" * 60)

teacher = TeacherNet().to(DEVICE)
t_opt   = torch.optim.Adam(teacher.parameters(), lr=1e-3)
scaler  = torch.amp.GradScaler("cuda", enabled=USE_AMP)

for epoch in range(3):
    teacher.train()
    running = 0.0
    for x, y in train_loader:
        x, y = x.to(DEVICE, non_blocking=True), y.to(DEVICE, non_blocking=True)
        t_opt.zero_grad(set_to_none=True)   # set_to_none avoids a memset, slightly faster
        with torch.amp.autocast("cuda", enabled=USE_AMP):
            logits = teacher(x)
            loss   = F.cross_entropy(logits, y)
        scaler.scale(loss).backward()
        scaler.step(t_opt)
        scaler.update()
        running += loss.item()
    print(f"  epoch {epoch+1}/3 | loss {running/len(train_loader):.4f}")

teacher_acc = accuracy(teacher, test_loader)
print(f"\nTeacher accuracy: {teacher_acc:.2f}%  ({count_params(teacher):,} params)")

# Freeze teacher — it will never be updated again.
# The teacher is now a fixed function that maps inputs to soft distributions.
teacher.eval()
for p in teacher.parameters():
    p.requires_grad_(False)


# ---------------------------------------------------------------------------
# SECTION 2 — Train student from scratch (no distillation, baseline)
#
# Cross-entropy on hard labels only. This is the control condition.
# ---------------------------------------------------------------------------

print("\n" + "=" * 60)
print("SECTION 2 — Train student (baseline, hard labels only)")
print("=" * 60)

student_baseline = StudentNet().to(DEVICE)
opt_baseline     = torch.optim.Adam(student_baseline.parameters(), lr=1e-3)

for epoch in range(3):
    student_baseline.train()
    running = 0.0
    for x, y in train_loader:
        x, y = x.to(DEVICE, non_blocking=True), y.to(DEVICE, non_blocking=True)
        opt_baseline.zero_grad(set_to_none=True)
        logits = student_baseline(x)
        loss   = F.cross_entropy(logits, y)
        loss.backward()
        opt_baseline.step()
        running += loss.item()
    print(f"  epoch {epoch+1}/3 | loss {running/len(train_loader):.4f}")

baseline_acc = accuracy(student_baseline, test_loader)
print(f"\nStudent (baseline) accuracy: {baseline_acc:.2f}%  ({count_params(student_baseline):,} params)")


# ---------------------------------------------------------------------------
# SECTION 3 — Train student with Knowledge Distillation
#
# The KD loss combines two terms:
#
#   L = α · CE(s_logits, hard_labels)
#     + (1 − α) · KL(softmax(s_logits / T) ‖ softmax(t_logits / T)) · T²
#
# TERM 1 — Cross-entropy on hard labels (standard supervised loss).
#
# TERM 2 — KL divergence between student and teacher soft distributions.
#
#   KL(P ‖ Q) = Σ P(i) · log(P(i) / Q(i))
#
#   P = teacher soft targets at temperature T
#   Q = student soft targets at temperature T
#
#   PyTorch's F.kl_div expects log-probabilities for Q:
#     KL(P ‖ Q) = Σ P(i) · (log P(i) − log Q(i))
#               = F.kl_div(log_Q, P, reduction="batchmean")
#   so we pass log_softmax for the student and softmax for the teacher.
#
# WHY TEMPERATURE?
#   Softmax concentrates probability mass — at T=1 the teacher's output might
#   be [0.001, 0.001, 0.003, 0.92, 0.002, ...] for an obvious "3". Almost all
#   information is in the top class; the soft label is barely different from hard.
#   Dividing logits by T > 1 before softmax flattens the distribution:
#   [0.08, 0.08, 0.10, 0.45, 0.09, ...] — now the inter-class relationships are
#   visible and the student can learn "3 is more like 8 than like 0".
#
#   Multiplying the KD term by T² compensates for the magnitude reduction
#   caused by dividing by T before the softmax. Without this, the KD term
#   would shrink relative to the CE term as T grows, making α unintuitive.
#   (The T² factor comes from the gradient analysis of the KL term — Hinton
#   et al. 2015 appendix derives it.)
#
# WHY α?
#   In practice α = 0.1 to 0.5 works well. Higher α puts more weight on ground
#   truth (useful when the teacher is imperfect); lower α trusts the teacher more.
# ---------------------------------------------------------------------------

print("\n" + "=" * 60)
print("SECTION 3 — Train student with Knowledge Distillation")
print("=" * 60)

T     = 5.0     # temperature — softens both distributions
ALPHA = 0.5     # weight on hard-label CE loss

student_kd = StudentNet().to(DEVICE)
opt_kd     = torch.optim.Adam(student_kd.parameters(), lr=1e-3)
scaler_kd  = torch.amp.GradScaler("cuda", enabled=USE_AMP)

for epoch in range(3):
    student_kd.train()
    running = 0.0
    for x, y in train_loader:
        x, y = x.to(DEVICE, non_blocking=True), y.to(DEVICE, non_blocking=True)

        # Teacher forward — no grad needed (frozen), just get soft targets
        with torch.no_grad():
            t_logits = teacher(x)
            t_probs  = F.softmax(t_logits / T, dim=1)   # soft distribution at temperature T

        opt_kd.zero_grad(set_to_none=True)
        with torch.amp.autocast("cuda", enabled=USE_AMP):
            s_logits     = student_kd(x)
            s_log_probs  = F.log_softmax(s_logits / T, dim=1)   # log Q for kl_div

            # KL divergence: how far is the student from the teacher at temperature T?
            kd_loss = F.kl_div(s_log_probs, t_probs, reduction="batchmean") * (T * T)
            # CE loss: keep the student honest against ground truth
            ce_loss = F.cross_entropy(s_logits, y)
            # Combined loss
            loss = ALPHA * ce_loss + (1 - ALPHA) * kd_loss

        scaler_kd.scale(loss).backward()
        scaler_kd.step(opt_kd)
        scaler_kd.update()
        running += loss.item()

    kd_acc = accuracy(student_kd, test_loader)
    print(f"  epoch {epoch+1}/3 | loss {running/len(train_loader):.4f} | test acc {kd_acc:.2f}%")

kd_acc = accuracy(student_kd, test_loader)
print(f"\nStudent (KD) accuracy: {kd_acc:.2f}%  ({count_params(student_kd):,} params)")


# ---------------------------------------------------------------------------
# SECTION 4 — Results
# ---------------------------------------------------------------------------

print("\n" + "=" * 60)
print("SECTION 4 — Results")
print("=" * 60)

t_params  = count_params(teacher)
s_params  = count_params(student_kd)
t_lat     = latency_ms(teacher,          test_loader)
b_lat     = latency_ms(student_baseline, test_loader)
kd_lat    = latency_ms(student_kd,       test_loader)

print(f"\n{'Model':30} {'Params':>10} {'Accuracy':>10} {'Latency ms/batch':>18}")
print("-" * 72)
print(f"{'Teacher (CNN)':30} {t_params:>10,} {teacher_acc:>9.2f}% {t_lat:>18.1f}")
print(f"{'Student baseline (hard labels)':30} {s_params:>10,} {baseline_acc:>9.2f}% {b_lat:>18.1f}")
print(f"{'Student KD (soft labels)':30} {s_params:>10,} {kd_acc:>9.2f}% {kd_lat:>18.1f}")
print()
print(f"Compression ratio:   {t_params / s_params:.0f}× fewer parameters")
print(f"Accuracy recovered:  {(kd_acc - baseline_acc):.1f} pp gain over baseline student (distillation benefit)")
print(f"Speed vs teacher:    {t_lat / kd_lat:.1f}× faster inference")
