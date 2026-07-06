# Lesson 4.1a — Pruning: Removing Redundant Weights

> **Lesson 4.1 / Sub-demo 1** — The simplest compression technique: identify weights that contribute little and zero them out. Two variants with fundamentally different hardware implications.

| | |
|---|---|
| **Problem this solves** | A trained model contains millions of near-zero weights that contribute almost nothing to predictions but consume memory and compute on every inference. Pruning identifies and removes them. |
| **Mental model** | Unstructured pruning: scatter zeros anywhere in the weight matrices — maximum flexibility, but the matrix is still the same shape so hardware sees no speedup without sparse kernels. Structured pruning: zero entire rows or neurons — the matrix physically shrinks, regular dense hardware gets faster. |
| **What the lecture demonstrates** | `unstructured_prune.ipynb`: global L1 magnitude pruning at 80% sparsity + CSR sparse storage. `structured_prune.ipynb`: ln_structured pruning at 30% + layer compaction to remove zeroed neurons entirely. |
| **Where this fits** | First step in the compression stack — reduces parameter count before quantization further reduces precision. |

---

# Model Compression & Pruning — Concepts & Guide

Demos: `unstructured_prune.ipynb` — global magnitude pruning at 80% + CSR sparse storage on MNIST MLP. `structured_prune.ipynb` — structured neuron pruning at 30% + layer compaction.

---

## Part 1 — Why Compression Matters

### The deployment gap

Training and deployment have opposite constraints.

During training, you want the largest model that fits on your GPU cluster — more
parameters, more capacity, better accuracy. Memory and speed are secondary concerns.

During deployment, the same model must run on hardware that may be a fraction of
that capacity: a mobile device, an edge sensor, a CPU-only inference server, or
a cloud function billed per millisecond.

The gap between "best training accuracy" and "deployable in production" is the
**deployment gap**. Model compression is the set of techniques that close it.

---

### What compression actually achieves

```
Goal              Technique              Gain
────────────────────────────────────────────────────────────
Smaller file      Unstructured pruning + CSR     ~60% size reduction
Faster inference  Structured pruning + compaction fewer FLOPs
Less memory       Quantization (INT8/FP16)        2–4× memory reduction
Faster + smaller  Knowledge distillation          train small model from large
```

This demo covers the first two: **unstructured pruning** with sparse storage and
**structured pruning** with layer compaction. Both are demonstrated on an MLP
trained on MNIST, which is small enough to iterate on quickly but large enough
to show the tradeoffs clearly.

---

### Where compression fits in the MLOps progression

```
1. Train to convergence       maximize accuracy, ignore size
2. Profile                    measure where time and memory go
3. Pruning                    remove weights or neurons            ← you are here
4. Quantization               reduce weight precision (FP32→INT8)
5. Knowledge distillation     train compact model from large one
6. Hardware-specific export   TensorRT, CoreML, ONNX, TFLite
```

Compression is not a one-time step. Real pipelines iterate: prune → fine-tune →
quantize → export → benchmark on target hardware → repeat.

---

## Part 2 — Unstructured Pruning

### The core idea

A neural network has millions of weights. Many of them are close to zero — they
contribute almost nothing to the output, but they still consume memory and
participate in every matrix multiplication.

**Unstructured pruning** removes the smallest weights (by absolute value) and sets
them to exactly zero. The result is a sparse weight matrix: most entries are zero,
a fraction are non-zero.

```
Before:  [ 0.03  -0.91   0.01   0.74  -0.02   0.58 ]  all weights active
After:   [ 0.00  -0.91   0.00   0.74   0.00   0.58 ]  small weights zeroed
```

The pattern of zeros has no structure — any weight can be zeroed regardless of
which neuron or layer it belongs to. This is what "unstructured" means.

---

### Global vs layer-wise pruning

The demo uses `prune.global_unstructured` — a single global threshold applied
across all Linear layers simultaneously:

```python
prune.global_unstructured(
    parameters_to_prune,
    pruning_method=prune.L1Unstructured,
    amount=0.80
)
```

Global pruning finds the bottom 80% of weights by magnitude across the entire
network and zeros them all. The distribution is uneven by design:

```
Layer fc1: sparsity = 0.841  (early layers tend to be pruned more)
Layer fc2: sparsity = 0.563
Layer fc3: sparsity = 0.362  (output layer pruned least)
Overall:   sparsity = 0.800
```

The output layer retains more weights because it has far fewer parameters — the
global threshold leaves it relatively intact. This is the right behavior: the
classifier head matters more per weight than the early feature layers.

Layer-wise pruning instead applies a fixed ratio per layer independently.
Global pruning tends to produce better accuracy because it lets the network
decide where redundancy actually lives.

---

### The reparametrization trick

PyTorch's pruning API does not immediately zero the weight tensor. Instead it
applies a **reparametrization**: the original weights are stored in `weight_orig`
and a binary mask is stored in `weight_mask`. The actual `weight` is computed as
their element-wise product at each forward pass.

```
weight = weight_orig * weight_mask
```

This means the pruning is reversible and the mask can be updated during fine-tuning.
The gradient flows through `weight_orig`; the masked positions receive gradients
but their updates are zeroed out by the mask on the next forward pass.

To convert the reparametrization into a real sparse tensor (and remove the overhead),
call `prune.remove`:

```python
for m in model.modules():
    if isinstance(m, nn.Linear):
        prune.remove(m, 'weight')
```

After this, `weight_orig` and `weight_mask` are gone. The `weight` tensor contains
real zeros. The model is smaller and has no extra parameters.

---

### Accuracy after pruning

```
Baseline (1 epoch):               test acc 0.9638
Immediately after 80% pruning:    test acc 0.9564  (−0.0074)
After 1 fine-tuning epoch:        test acc 0.9717  (+0.0079 vs baseline)
```

Two things to notice:

1. **Accuracy drop is modest at 80% sparsity.** Most of the pruned weights were
   genuinely redundant. The network carries significant slack capacity.

2. **Fine-tuning recovers and slightly exceeds baseline.** The network is smaller
   but the surviving weights are trained more effectively once the dead weights
   are gone. This is a common pattern — brief fine-tuning at a lower learning
   rate (`5e-4` vs original `1e-3`) almost always recovers pruning losses.

---

### The critical caveat: pruning alone saves nothing

This is the most important thing to understand about unstructured pruning.

After pruning and `prune.remove`, the weight tensors are the same size and shape
as before. The zeros are stored as regular float32 values. Dense matrix multiplication
(`nn.Linear`) computes every element — it does not skip zeros.

```
Dense model after pruning: 0.90 MB, same inference time as before
```

To actually benefit from the sparsity, you need a different storage format.

---

### CSR: converting sparsity into real memory savings

**Compressed Sparse Row (CSR)** is a standard format for sparse matrices.
Instead of storing every element, it stores only the non-zero values plus
two index arrays that encode their positions.

```python
from scipy.sparse import csr_matrix

csr_W = csr_matrix(W)    # W is a dense numpy array
# csr_W.data      — non-zero values only
# csr_W.indices   — column index of each value
# csr_W.indptr    — row boundaries
```

On an 80%-sparse matrix, roughly 80% of the entries are zero and don't need
to be stored. The result on this model:

```
Dense storage:  0.90 MB
CSR storage:    0.36 MB
Memory saving:  59.8%
```

The trade-off is that standard PyTorch inference cannot use CSR weights directly.
CSR is appropriate for:
- Saving the model to disk (serialization)
- Deploying to hardware with sparse-aware kernels (NVIDIA cuSPARSE, Ampere 2:4 sparsity)
- Serving with sparse matrix libraries

---

## Part 3 — Structured Pruning

### Why structured pruning exists

Unstructured pruning produces weights with an arbitrary zero pattern. Hardware
and software that accelerate dense matrix multiplication cannot exploit this — you
need sparse-aware kernels, which are not universally available.

**Structured pruning** removes entire structural units: rows, columns, or channels.
The result is a smaller dense matrix, not a sparse one. Standard dense operations
are immediately faster on the smaller matrix, on any hardware.

```
Unstructured: zero individual weights   → sparse matrix (needs sparse kernels)
Structured:   zero entire rows/columns  → smaller dense matrix (works everywhere)
```

---

### Neuron pruning: removing rows from a weight matrix

In a fully-connected layer, each output neuron corresponds to one row of the weight
matrix. Pruning a row effectively removes that neuron: it no longer produces an
activation that any downstream layer can use.

The demo uses `ln_structured` to prune 30% of the neurons in `fc2` (L2-norm ranking):

```python
prune.ln_structured(model.fc2, name="weight", amount=0.30, n=2, dim=0)
```

`dim=0` means "prune along rows" (output neurons). `n=2` uses the L2-norm to score
each row — rows with the smallest norm are the least informative and are removed first.

```
fc2 neurons kept: 90 / 128   (30% removed)

Immediately after pruning:    acc 0.9637  (baseline was 0.9638 — almost no drop)
After 1 fine-tuning epoch:    acc 0.9725
```

The minimal accuracy drop at 30% removal shows that a large fraction of the neurons
were redundant for this task.

---

### Layer compaction: making the savings real

The same caveat applies: pruning rows still leaves a matrix with the same shape
— the pruned rows are zeros, not gone. To actually reduce computation, the layer
must be **compacted**: rebuilt with fewer neurons.

```python
def compact_fc2_fc3(model, keep_mask_bool):
    keep_idx = keep_mask_bool.nonzero(as_tuple=False).squeeze(1)
    new_out = keep_idx.numel()

    new_fc2 = nn.Linear(old_fc2.in_features, new_out)
    new_fc3 = nn.Linear(new_out, old_fc3.out_features)

    new_fc2.weight.copy_(old_fc2.weight[keep_idx])       # keep surviving rows
    new_fc2.bias.copy_(old_fc2.bias[keep_idx])
    new_fc3.weight.copy_(old_fc3.weight[:, keep_idx])    # keep corresponding columns
    new_fc3.bias.copy_(old_fc3.bias)

    model.fc2, model.fc3 = new_fc2, new_fc3
```

This rebuilds `fc2` with 90 output neurons instead of 128, and `fc3` with 90 input
neurons instead of 128. The compacted network is a fully dense, smaller model:

```
Before compaction:  235,146 parameters  |  file: 921.7 KB
After compaction:   225,000 parameters  |  file: 882.0 KB
Test accuracy:      0.9725  (unchanged — compaction is lossless)
```

Compaction is the critical step that turns structured pruning into real inference
speedup. Without it, you have a sparse matrix that happens to have dense structure —
but the compute graph is identical to the original.

---

## Part 4 — Unstructured vs Structured: When to Use Which

| | Unstructured | Structured |
|---|---|---|
| Granularity | Individual weights | Entire neurons / channels / filters |
| Accuracy impact | Lower (finer control) | Higher (coarser removal) |
| Memory saving | Needs sparse format (CSR, CUDA sparse) | Immediate — smaller dense tensors |
| Inference speedup | Only with sparse-aware hardware | Universal — standard dense matmul |
| Best for | Serialization, sparse hardware (Ampere) | Edge deployment, CPU, any hardware |
| Typical sparsity | 50–90% | 10–50% |

The two approaches are complementary. A common production pipeline:

```
1. Structured pruning + compaction   → smaller dense model
2. Unstructured pruning on compact   → additional sparsity
3. INT8 quantization                 → 4× memory reduction on top
4. Export to ONNX / TensorRT         → hardware-optimized inference
```

---

## Part 5 — The Fine-Tuning Protocol

Both demos use the same fine-tuning pattern:
1. Train baseline at `lr=1e-3`
2. Apply pruning (masks in place, reparametrization active)
3. Fine-tune at `lr=5e-4` (half the original — surviving weights are already good)
4. Make pruning permanent with `prune.remove`

Why lower learning rate? The surviving weights are already near a good local minimum.
A large learning rate would overshoot it. A smaller rate makes small corrections
to compensate for the removed weights.

The number of fine-tuning epochs here is 1 (kept minimal for speed). In production,
iterative pruning — prune a little, fine-tune, prune more, fine-tune — produces
higher accuracy at a given sparsity level than one-shot pruning.

---

## Quick Reference

### Unstructured pruning

```python
from torch.nn.utils import prune

# Collect parameters to prune
params = [(m, 'weight') for m in model.modules() if isinstance(m, nn.Linear)]

# Prune globally at 80%
prune.global_unstructured(params, pruning_method=prune.L1Unstructured, amount=0.80)

# Fine-tune, then make permanent
prune.remove(m, 'weight')  # run for each module
```

### Structured pruning (neuron)

```python
# Remove 30% of output neurons in a layer by L2-norm
prune.ln_structured(model.fc2, name="weight", amount=0.30, n=2, dim=0)

# Make permanent, then compact the layer
prune.remove(model.fc2, 'weight')
# Rebuild fc2 and fc3 with keep_idx rows/columns
```

### CSR sparse storage

```python
from scipy.sparse import csr_matrix
import pickle

csr_weights = {}
for name, m in model.named_modules():
    if isinstance(m, nn.Linear):
        csr_weights[name] = csr_matrix(m.weight.detach().cpu().numpy())

with open("model_pruned_csr.pkl", "wb") as f:
    pickle.dump(csr_weights, f)
```

### Measure sparsity

```python
def sparsity(tensor):
    return (tensor == 0).sum().item() / tensor.numel()

for name, m in model.named_modules():
    if isinstance(m, nn.Linear):
        print(f"{name}: {sparsity(m.weight.data):.1%} sparse")
```

---

## Official Documentation

- PyTorch pruning tutorial: https://pytorch.org/tutorials/intermediate/pruning_tutorial.html
- `torch.nn.utils.prune`: https://pytorch.org/docs/stable/nn.html#utilities
- scipy CSR format: https://docs.scipy.org/doc/scipy/reference/generated/scipy.sparse.csr_matrix.html
- NVIDIA structured sparsity (Ampere 2:4): https://developer.nvidia.com/blog/accelerating-inference-with-sparsity-using-ampere-and-tensorrt/
