# Lesson 4.1c — Knowledge Distillation

> **Lesson 4.1 / Sub-demo 3** — When pruning and quantization aren't enough, train a smaller model that mimics a larger one. The key insight: the teacher's output distribution contains more information than hard labels alone.

| | |
|---|---|
| **Problem this solves** | Some models are architecturally too large to prune or quantize to the target size without unacceptable accuracy loss. A fundamentally smaller architecture is needed — but training it from scratch on hard labels leaves accuracy on the table. |
| **Mental model** | Hard labels say "this is a 3." The teacher says "this is 87% a 3, 9% an 8, 3% a 2." That inter-class structure is generalizable knowledge — and the student can learn it even if it can't replicate the teacher's capacity. |
| **What the lecture demonstrates** | Train a CNN teacher (1.2M params) → train a student MLP (16K params) with and without distillation → compare: params, accuracy, inference speed. |
| **Where this fits** | Optional third step in the compression stack — use when pruning + quantization alone can't hit the deployment target. |

---

# Knowledge Distillation — Concepts & Guide

Demo: `kd.py` — training a tiny MLP student (16 K params) from a CNN teacher (1.2 M params) on MNIST using soft-target distillation. 75× compression with near-teacher accuracy. Run: `uv run --no-sync python kd.py`

---

## Part 1 — Why Knowledge Distillation

### The deployment gap

The same gap covered in pruning applies here, but from a different angle.

Pruning starts with a large model and removes parts of it. Knowledge distillation takes
a different path: you keep the large model unchanged, and you use it to teach a smaller
model that was designed from the start to be cheap to run.

The key insight is that a trained model carries more information than the labels alone.
When a teacher model outputs `[0.02, 0.01, 0.94, 0.01, 0.01, ...]` for a handwritten
"2", the small non-zero probabilities on "3" and "7" encode the teacher's understanding
of which digits are similar. Raw labels (`[0, 0, 1, 0, 0, ...]`) throw that away.

Distillation preserves it.

---

### What distillation actually achieves

```
Goal                    Technique                           Gain
──────────────────────────────────────────────────────────────────────
Smaller model           Train student from teacher outputs  design-time param reduction
Faster inference        Smaller architecture                fewer FLOPs per forward pass
Comparable accuracy     Soft targets + temperature          recovers gap vs training alone
Label efficiency        Teacher supervises student          works with less labelled data
```

This demo covers the standard offline distillation setup: train the teacher to
convergence first, freeze it, then train the student with a combined loss.

---

### Where KD fits in the MLOps progression

```
1. Train to convergence       maximize accuracy, ignore size
2. Profile                    measure where time and memory go
3. Pruning                    remove weights or neurons
4. Quantization               reduce weight precision (FP32→INT8)
5. Knowledge distillation     train compact model from large one    ← you are here
6. Hardware-specific export   TensorRT, CoreML, ONNX, TFLite
```

KD is orthogonal to pruning and quantization — a distilled student can be pruned
and quantized further. It is typically applied when you want to design a permanently
smaller model rather than compress an existing one post-hoc.

---

## Part 2 — The Distillation Loss

### Two supervision signals

Training the student with hard labels alone (one-hot targets) produces a model that
knows the right answer but not the teacher's confidence structure. The distillation
loss adds a second term that aligns the student's output distribution with the
teacher's.

```
L = α · CE(student_logits, hard_labels)
  + (1 − α) · KL(softmax(student/T) ∥ softmax(teacher/T)) · T²
```

- **CE** — standard cross-entropy against ground-truth labels
- **KL** — KL divergence between the student's and teacher's softened distributions
- **α** — mixing weight; `0.5` balances both signals equally
- **T** — temperature; controls how soft the distributions are
- **T²** — scaling factor that compensates for the gradient magnitude reduction
           introduced by dividing logits by T before softmax

The demo uses `T=5.0`, `α=0.5`.

---

### Temperature scaling

Without temperature, the teacher's softmax is already nearly one-hot for a well-trained
model — the largest logit dominates. This gives the student almost the same information
as the hard label.

Temperature `T` divides the logits before softmax, flattening the distribution:

```
T = 1  (no scaling):  [ 0.0001  0.0001  0.9996  0.0001  0.0001 ]  nearly hard
T = 5  (demo):        [ 0.08    0.07    0.57    0.12    0.09   ]  soft, informative
T = 10 (very soft):   [ 0.15    0.14    0.28    0.22    0.17   ]  nearly uniform
```

Higher T reveals inter-class similarity (which digits look like which) but at
some point the signal becomes too diffuse to guide learning. Values between 3 and 10
are common; `T=5` is a good starting point.

Both teacher and student are divided by the same T before computing the KL term.
The T² factor restores the gradient scale to match the CE term.

---

### The α trade-off

`α` controls how much the student learns from hard labels vs. soft teacher targets.

```
α = 1.0   student ignores teacher, trains on labels only  (baseline, no distillation)
α = 0.5   equal weight on both signals                    (demo)
α = 0.0   student ignores labels, trains on teacher only  (pure distillation)
```

In practice, some label signal (`α > 0`) prevents the student from drifting if the
teacher is wrong on some examples. `α = 0.5` is the standard default.

---

## Part 3 — Teacher and Student Architecture

### Teacher: CNN (~1.2 M parameters)

```
Input (1×28×28)
  → Conv2d(1, 32, 3)   → ReLU
  → Conv2d(32, 64, 3)  → ReLU
  → MaxPool2d(2)
  → Flatten  →  FC(9216 → 128)  → ReLU
              →  FC(128 → 10)
```

Three epochs on MNIST: **98.73% test accuracy**.

The teacher is a modest CNN — deeper or wider architectures would push accuracy
further, but 98.7% is sufficient to supervise a student here.

---

### Student: MLP (~16 K parameters)

```
Input (1×28×28)
  → Flatten
  → FC(784 → 20)  → ReLU
  → FC(20 → 10)
```

The student has no convolutions and only 20 hidden neurons. Trained alone on hard
labels, it would struggle. With distillation from the teacher it reaches **93.3%**
in 3 epochs — a level that would require many more epochs or a larger architecture
without the teacher's guidance.

```
Params — Teacher: 1,199,882 | Student: 15,910 | Compression: 75.4×
```

---

## Part 4 — Training Protocol

### Step 1: Train the teacher to convergence

```python
teacher = TeacherNet().to(device)
opt = torch.optim.Adam(teacher.parameters(), lr=1e-3)

# train with standard cross-entropy
loss = F.cross_entropy(teacher(x), y)
```

The teacher is trained until accuracy plateaus. In the demo, 3 epochs reach 98.73%.
In production, train the teacher fully — the stronger the teacher, the more information
it can pass on.

---

### Step 2: Freeze the teacher

```python
teacher.eval()
for p in teacher.parameters():
    p.requires_grad_(False)
```

The teacher is frozen before distillation begins. Its weights must not update — the
student is learning from a fixed reference. Forgetting to freeze the teacher causes
both models to co-adapt, which destroys the distillation signal.

---

### Step 3: Train the student with the combined loss

```python
with torch.no_grad():
    t_logits = teacher(x)
    t_probs  = F.softmax(t_logits / T, dim=1)   # soft targets

s_logits    = student(x)
s_log_probs = F.log_softmax(s_logits / T, dim=1)

kd_loss = F.kl_div(s_log_probs, t_probs, reduction="batchmean") * (T * T)
ce_loss = F.cross_entropy(s_logits, y)
loss    = ALPHA * ce_loss + (1 - ALPHA) * kd_loss
```

Note `torch.no_grad()` around the teacher forward pass — no gradient needed there,
and it reduces memory usage.

---

### Results from the demo

```
Epoch 1:  loss 4.3512 | test acc 89.98%
Epoch 2:  loss 2.1457 | test acc 92.05%
Epoch 3:  loss 1.7205 | test acc 93.32%

Teacher test acc:       98.73%
Student (KD) test acc:  93.32%
Accuracy gap:           5.4 pp
Compression:            75.4×
```

The student closes most of the gap to the teacher in 3 epochs. A student trained on
hard labels alone at this size would plateau around 87–89%; distillation adds roughly
4–6 percentage points.

---

## Part 5 — KD vs Pruning: When to Use Which

| | Pruning | Knowledge Distillation |
|---|---|---|
| Starting point | Existing large model | Existing large model (teacher) |
| Output | Same architecture, fewer weights | New smaller architecture |
| Accuracy recovery | Fine-tuning after pruning | Baked into the distillation training |
| Architecture freedom | None — topology is fixed | Full — student can be any design |
| Implementation effort | Low (PyTorch prune API) | Medium (custom loss, two-model training) |
| Best for | Reducing an already-trained model | Designing a production-optimised model |
| Stackable | Yes — prune the student further | Yes — quantize or prune the student |

The two techniques are complementary. A common production pipeline:

```
1. Train large teacher to full accuracy
2. Design student architecture for target hardware
3. Distillation training                     → compact student, most accuracy recovered
4. Structured pruning on student             → additional neuron removal
5. INT8 quantization                         → 4× memory reduction on top
6. Export to ONNX / TensorRT                 → hardware-optimised inference
```

---

## Quick Reference

### Distillation loss

```python
T = 5.0
ALPHA = 0.5

with torch.no_grad():
    t_probs = F.softmax(teacher(x) / T, dim=1)

s_log_probs = F.log_softmax(student(x) / T, dim=1)
kd_loss = F.kl_div(s_log_probs, t_probs, reduction="batchmean") * (T ** 2)
ce_loss = F.cross_entropy(student(x), y)
loss    = ALPHA * ce_loss + (1 - ALPHA) * kd_loss
```

### Freeze teacher before distillation

```python
teacher.eval()
for p in teacher.parameters():
    p.requires_grad_(False)
```

### Count parameters

```python
def count_params(model):
    return sum(p.numel() for p in model.parameters())

print(f"Teacher: {count_params(teacher):,} | Student: {count_params(student):,}")
print(f"Compression: {count_params(teacher)/count_params(student):.1f}×")
```

---

## Official Documentation

- Hinton et al. — original KD paper: https://arxiv.org/abs/1503.02531
- PyTorch KL divergence loss: https://pytorch.org/docs/stable/generated/torch.nn.KLDivLoss.html
- PyTorch AMP (mixed precision used in demo): https://pytorch.org/docs/stable/amp.html
