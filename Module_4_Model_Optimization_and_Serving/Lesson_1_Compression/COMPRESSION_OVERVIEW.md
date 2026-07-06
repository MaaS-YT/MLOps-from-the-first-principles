# Lesson 4.1 — Model Compression and Deployment Optimization

> **Lesson 4.1** — Training optimizes for accuracy. Deployment optimizes for latency, memory, and cost. The gap between them is real in almost every production system. Pruning, quantization, distillation, and ONNX export are the tools that close it.

| | |
|---|---|
| **Problem this solves** | A model that achieves 94% accuracy on a data center GPU may require 400ms and 2GB RAM per inference. An edge device has 50ms and 256MB. The model that trained cannot be the model that deploys — compression is the engineering work that bridges them. |
| **Mental model** | Compression is a layered stack, not a single technique: prune redundant weights → quantize remaining weights to lower precision → optionally replace with a compact distilled model → export to ONNX for hardware-agnostic deployment. Each layer is independent and additive. |
| **What the lecture demonstrates** | Sub-demo 1 (pruning): unstructured and structured pruning on an MNIST MLP — what sparsity means and when it actually saves compute. Sub-demo 2 (quantization): INT8 post-training quantization with calibration. Sub-demo 3 (KD): teacher-student training with soft labels. Sub-demo 4 (ONNX): exporting to ONNX, inspecting the protobuf graph, and benchmarking ORT optimization levels. |
| **Where this fits** | This module adds the **Optimization Layer** — the last step before a model reaches a production deployment target. The output of this module is an ONNX file that any inference runtime can consume. |

---

# Model Compression & Optimization — Concepts & Overview

Sub-demos: `pruning/` — unstructured and structured pruning on MNIST MLP. `Quantization/` — dynamic and static INT8 PTQ. `KD/` — teacher-student training with soft labels. `onnx/` — ONNX export, graph inspection, ORT benchmarking.

---

## The Problem This Section Solves

Training optimizes for one thing: accuracy. The training environment has access to
large GPUs, long compute budgets, and no hard constraint on model size. The result
is typically a large, slow, memory-hungry model that achieves the best possible score
on the validation set.

Deployment optimizes for completely different things: latency, memory footprint,
energy consumption, and cost per inference. A model that takes 400ms to run on a
data center GPU may need to run in 10ms on a mobile CPU.

This gap between "best training accuracy" and "deployable in production" is real
in almost every production ML system. Model compression is the set of techniques
that close it — without retraining from scratch.

---

## The Three Approaches

There are three main families of compression techniques. Each attacks the problem
from a different angle, and each has a different mental model.

---

### Pruning — Remove what isn't needed

The core insight behind pruning: **trained neural networks are massively over-parameterized**.
A network trained on MNIST has far more weights than the task mathematically requires.
Many weights contribute almost nothing to the output — their values are near zero,
and zeroing them completely changes the predictions by a negligible amount.

Pruning exploits this by identifying and removing the least important weights.

The mental model is a **sculpting process**:

```
Trained model (dense)
  │
  │  identify least important weights (by magnitude, gradient, etc.)
  │  set them to zero
  ▼
Sparse model (same shape, mostly zeros)
  │
  │  fine-tune briefly to recover accuracy
  │  convert sparse storage / compact the architecture
  ▼
Compressed model (smaller, faster)
```

Two distinct strategies:

**Unstructured pruning** zeros individual weights anywhere in the network — the
smallest values regardless of which layer or neuron they belong to. The result is
a sparse weight matrix. The network architecture is unchanged; the tensors are the
same size. To benefit from the sparsity, you need sparse-aware storage (CSR) or
hardware that skips zero multiplications (NVIDIA Ampere 2:4 sparsity).

**Structured pruning** zeros entire structural units — rows in a weight matrix
(neurons), channels in a convolution (filters). The result is a smaller dense matrix.
Because the architecture actually shrinks, any hardware running standard dense
matrix multiplication immediately benefits — no sparse kernels required.

The tradeoff: unstructured pruning causes less accuracy loss at a given compression
ratio because it has finer control. Structured pruning causes more accuracy loss but
produces a model that is universally faster.

---

### Quantization — Reduce precision

The core insight behind quantization: **weights don't need 32 bits**.

A standard trained model stores every weight as a 32-bit floating point number.
That's 4 bytes per weight. An INT8 representation uses 1 byte per weight — the same
information, stored in one quarter of the space, with arithmetic that is 2–4× faster
on most hardware because integer operations are cheaper than floating point.

The mental model is a **lossy compression** applied to the numerical representation
of weights, not to the architecture:

```
Original weight:   0.7382619...   (float32, 32 bits, exact)
Quantized weight:  94              (int8, 8 bits, approximate)
                   → dequantized to 0.7344 when used in computation
```

The quantization process maps the continuous range of floating point values onto
a discrete integer grid. The mapping is learned from a small calibration dataset —
the system measures the actual range of activations during a few forward passes,
then chooses the integer grid that minimizes rounding error.

Three quantization modes:

**Post-training quantization (PTQ)**: quantize a trained model without any
retraining. Fast to apply, causes modest accuracy loss, works well when the model
is already well-trained and the calibration dataset is representative.

**Quantization-aware training (QAT)**: simulate quantization during training by
inserting fake quantization operators into the forward pass. The model learns
weight values that are robust to the rounding that will happen at inference time.
More expensive, produces better accuracy.

**Dynamic quantization**: quantize weights statically but quantize activations
dynamically at inference time. Good for models with variable-length inputs
(transformers, RNNs) where activation ranges vary per input.

Quantization is the highest-leverage single technique in practice. A 4× memory
reduction with negligible accuracy loss is common on standard models.

---

### Knowledge Distillation — Transfer knowledge, not weights

Pruning and quantization both start from a trained model and make it smaller.
Distillation takes a different approach: **train a small model to mimic a large one**.

The core insight: a large, accurate model has learned something that is encoded
in its output probability distributions, not just in its final predictions.
When a classifier predicts "cat: 0.85, lynx: 0.12, dog: 0.03", the small
probabilities for related classes carry information about the structure of the
problem — that cats and lynxes are similar, that cats and dogs are less similar.
Hard labels ("cat: 1, everything else: 0") discard this structure.

Distillation transfers this richer signal to a smaller model:

```
Teacher model (large, accurate, slow)
  │
  │  run on training data → soft probability distributions
  │  ("soft labels": probabilities across all classes, not just the winner)
  ▼
Student model (small, fast)
  │  trained to match teacher's soft outputs, not just ground-truth labels
  │  learns not just "what is correct" but "how similar are the classes"
  ▼
Deployed model (student only — teacher is discarded)
```

The student is trained on a combination of two losses:

```
Total loss = α × cross_entropy(student_output, ground_truth)
           + (1−α) × KL_divergence(student_output, teacher_output)
```

The second term — matching the teacher's soft distribution — provides a richer
gradient signal than ground-truth labels alone. The student often achieves accuracy
close to the teacher at a fraction of the size.

The mental model is **apprenticeship**: the student doesn't just learn the right
answers, it learns the teacher's reasoning about the problem. A student trained
only on correct answers may learn to be correct; a student trained on the teacher's
full distribution learns to be correct in the same *way* as the teacher.

---

## How the Three Techniques Relate

They are not alternatives — they are layers of a compression stack. In practice
they are applied in sequence:

```
Trained model (full precision, full size)
  │
  │  1. Pruning
  │     Remove redundant weights or neurons
  │     → smaller architecture (structured) or sparse tensors (unstructured)
  │
  ▼
Pruned model
  │
  │  2. Quantization
  │     Reduce weight precision from FP32 to INT8
  │     → 4× memory reduction, 2–4× arithmetic speedup
  │
  ▼
Quantized model
  │
  │  3. Export (TensorRT, ONNX, CoreML, TFLite)
  │     Compile for target hardware
  │
  ▼
Production model (edge / mobile / server)
```

Distillation sits outside this stack — it replaces the model rather than
compressing it. When the original model is too large to be a starting point
for pruning and quantization, distillation produces a better starting point.

```
Large teacher model
  │
  │  Distillation
  │
  ▼
Compact student model
  │
  │  Pruning + Quantization (optional further compression)
  │
  ▼
Production model
```

---

## Accuracy vs Compression: The Tradeoff Curve

Every compression technique trades accuracy for efficiency. The tradeoff is not
a fixed point — it is a curve controlled by a compression ratio parameter.

```
Accuracy
  │
  ●  Original model
  │
  ●  10% pruned / INT8 / small student    ← barely any loss
  │
  ●  50% pruned / INT4 / medium student
  │
  ●  90% pruned / INT2 / tiny student     ← significant degradation
  │
  └──────────────────────────────────────── Compression ratio
```

The practical question is always: **what accuracy loss is acceptable for the
deployment target?** A 1% accuracy drop that saves 60% memory and 3× latency
is worth it for most applications. A 5% drop may be acceptable for low-stakes
inference. A 0.1% drop may be unacceptable for medical or financial predictions.

The answer comes from measuring, not from theory. Every compression decision
should be validated on a held-out evaluation set against the deployment target.

---

## Where This Fits in the MLOps Progression

```
1. Train to convergence       maximize accuracy, ignore efficiency
2. Profile                    identify where memory and time go
3. Pruning                    remove redundant weights / neurons        ← pruning/
4. Quantization               reduce weight precision                   ← Quantization/
5. Distillation               replace with a compact trained model      ← KD/
6. Hardware export            TensorRT / CoreML / ONNX / TFLite
7. Inference monitoring       track latency and accuracy drift in prod
```

Steps 3–5 are not always all necessary. Start with quantization — it is the
least invasive and gives the largest single gain. Add pruning if further compression
is needed. Use distillation when the model is architecturally too large to prune
and quantize to the target.

---

## Sub-Demo Reference

| Demo | Technique | What it shows |
|---|---|---|
| `pruning/` — `PRUNING_GUIDE.md` + notebooks | Pruning | Unstructured (L1, 80%), structured (ln_structured, 30%), layer compaction, CSR storage |
| `Quantization/` — `qnt.py` + `QNT_GUIDE.md` | Quantization | FP32 baseline, dynamic INT8, static PTQ with calibration, latency comparison |
| `KD/` — `kd.py` + `KD_GUIDE.md` | Distillation | Teacher CNN → student MLP, soft labels, temperature scaling, KL divergence loss |
| `onnx/` — `onnx_export.py` + `ONNX_GUIDE.md` | ONNX export | Protobuf graph, JIT tracing, ORT optimization levels, EP benchmarking |

Each sub-demo has its own guide covering the tool-specific implementation details.
This document covers the concepts that apply to all three.
