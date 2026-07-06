# Lesson 4.1b — Quantization

> **Lesson 4.1 / Sub-demo 2** — Reduce the numerical precision of weights and activations from 32-bit float to 8-bit integer. Up to 4× smaller on disk, 2–4× faster on modern CPUs — with minimal accuracy loss on most tasks.

| | |
|---|---|
| **Problem this solves** | FP32 models store every weight as a 4-byte float. Most of that precision is wasted — the meaningful information is the weight's relative position within its range, not its absolute FP32 value. INT8 (256 levels) is sufficient for most tasks. |
| **Mental model** | Quantization is a change of representation, not a change of architecture. The same model, the same weights — stored and computed at lower precision. The mapping between FP32 and INT8 (scale + zero_point) is the only new information. |
| **What the lecture demonstrates** | FP32 baseline → dynamic INT8 (weights only, no calibration) → static PTQ (weights + activations, with calibration pass). Size, accuracy, and CPU latency compared across all three. |
| **Where this fits** | Second step in the compression stack — usually the highest single-step gain. Apply after pruning, before ONNX export. |

---

# Quantization — Concepts & Guide

Demo: `qnt.py` — FP32 MLP baseline trained on MNIST, then compressed with dynamic INT8 quantization and static post-training quantization (PTQ). Same 784→256→128→10 architecture used across all compression demos. Run: `uv run --no-sync python qnt.py`

---

## Part 1 — Why Quantization

### The deployment gap

Pruning removes weights. Distillation trains a smaller model. Quantization takes a third path: it keeps the same architecture and the same weights, but stores and computes them at lower numerical precision.

A standard PyTorch model uses 32-bit floats (FP32) for every weight and activation. Most of that precision is wasted for inference — the difference between `0.31415926` and `0.314` has no meaningful impact on a prediction. Quantization exploits this by rounding weights to 8-bit integers (INT8), reducing storage by 4× and unlocking integer arithmetic units that are significantly faster than floating-point units on CPUs and many edge accelerators.

---

### What quantization actually achieves

```
Goal                    Technique                    Gain
──────────────────────────────────────────────────────────────────────
Smaller model on disk   FP32 → INT8 weights          ~4× size reduction
Faster CPU inference    INT8 arithmetic              2–4× speedup on x86/ARM
Lower memory bandwidth  Smaller activations          fewer bytes moved per op
Edge deployment         INT8 / INT4 export           required by many accelerators
```

Quantization is typically the last compression step applied — after pruning or distillation — because it is precision-reducing and can amplify errors from an already-degraded model.

---

### Where quantization fits in the MLOps progression

```
1. Train to convergence       maximize accuracy, ignore size
2. Profile                    measure where time and memory go
3. Pruning                    remove weights or neurons
4. Knowledge distillation     train compact model from large one
5. Quantization               reduce weight precision              ← you are here
6. Hardware-specific export   TensorRT, CoreML, ONNX, TFLite
```

Quantization is the bridge between a compressed PyTorch model and hardware-optimised deployment. TensorRT, TFLite, and CoreML all operate primarily in INT8.

---

## Part 2 — FP32 vs INT8: What Changes

### Numerical representation

FP32 uses 32 bits to represent a number: 1 sign bit, 8 exponent bits, 23 mantissa bits.
The range is enormous (~±3.4 × 10³⁸) but most model weights cluster in a small range like `[-1, 1]`.

INT8 uses 8 bits to represent a signed integer in `[-128, 127]`. Mapping the weight range to this 256-step scale is the core operation of quantization:

```
w_int8 = round(w_fp32 / scale) + zero_point

scale     = (w_max - w_min) / 255
zero_point = round(-w_min / scale)
```

`scale` and `zero_point` are stored alongside the quantized weights and used to reconstruct an approximate FP32 value during inference:

```
w_fp32_approx = (w_int8 - zero_point) * scale
```

The approximation error is the quantization noise. For well-trained models it is small; accuracy drops of less than 1% are common with INT8.

---

### Per-tensor vs per-channel quantization

A single scale/zero_point for an entire weight matrix (`per-tensor`) is coarser but simpler.
One scale/zero_point per output channel (`per-channel`) matches the scale to the actual range of each filter, reducing quantization error significantly.

PyTorch's `fbgemm` backend (x86 CPUs) uses per-channel quantization for weights by default, which is why the accuracy drop is typically negligible.

---

## Part 3 — Dynamic Quantization

### How it works

Dynamic quantization quantizes the **weights** to INT8 at conversion time. The **activations** are quantized on-the-fly during each forward pass — the scale is computed from the actual activation values seen at runtime, not from a calibration dataset.

```python
model_dynamic = torch.quantization.quantize_dynamic(
    model_cpu, {nn.Linear}, dtype=torch.qint8
)
```

One line. No calibration. The original FP32 model is unchanged; a new quantized model is returned.

---

### What changes under the hood

Before:
```
x (FP32) → nn.Linear (FP32 weights, FP32 matmul) → y (FP32)
```

After dynamic quantization:
```
x (FP32) → quantize(x) → INT8 matmul (INT8 weights) → dequantize → y (FP32)
```

The weights are stored as INT8. At runtime, the activations are quantized to INT8 just before the matmul, then dequantized back to FP32 after. The output is still FP32.

---

### Trade-offs

- **Pro:** Zero calibration required — just call `quantize_dynamic` on any trained model.
- **Pro:** Weights are 4× smaller on disk; INT8 matmul is faster.
- **Con:** Activations are re-quantized every forward pass, adding overhead for small batches.
- **Con:** No activation quantization at rest — memory during inference is not reduced.

Dynamic quantization is the right choice for NLP models (LSTM, Transformer) and anywhere a calibration dataset is not available.

---

## Part 4 — Static Post-Training Quantization (PTQ)

### How it works

Static PTQ quantizes both weights **and** activations to INT8 ahead of time. Instead of computing activation scales at runtime, they are measured once during a **calibration pass** over a small representative dataset, then baked into the model.

The flow has three steps:

```
1. Prepare  — insert observers that record min/max of activations
2. Calibrate — run a subset of real data through the model; observers collect stats
3. Convert  — replace observers with fixed scale/zero_point; fold into INT8 ops
```

---

### QuantStub and DeQuantStub

PyTorch requires explicit markers for where quantization starts and ends. The model needs two wrappers:

```python
class QuantizableMLP(nn.Module):
    def __init__(self):
        self.quant   = torch.quantization.QuantStub()    # FP32 → INT8 boundary
        self.fc1     = nn.Linear(28*28, 256)
        self.fc2     = nn.Linear(256, 128)
        self.fc3     = nn.Linear(128, 10)
        self.relu    = nn.ReLU()
        self.dequant = torch.quantization.DeQuantStub()  # INT8 → FP32 boundary

    def forward(self, x):
        x = x.view(x.size(0), -1)
        x = self.quant(x)           # quantize input here
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        x = self.fc3(x)
        return self.dequant(x)      # dequantize output here
```

Everything between `quant` and `dequant` runs in INT8 after conversion. The FP32 weights from the original model are loaded with `strict=False` — the stubs have no pretrained weights.

---

### Calibration

```python
model_static.qconfig = torch.quantization.get_default_qconfig("fbgemm")
torch.quantization.prepare(model_static, inplace=True)   # insert observers

with torch.no_grad():
    for i, (x, _) in enumerate(calib_loader):
        model_static(x)
        if i >= 10:   # ~640 samples is sufficient for MNIST
            break

torch.quantization.convert(model_static, inplace=True)   # observers → INT8 ops
```

The calibration dataset should be representative of real inference data. For MNIST, 640 samples covers enough variation. For production models, a few hundred to a few thousand samples from the validation set is typical.

---

### Trade-offs vs dynamic

| | Dynamic | Static PTQ |
|---|---|---|
| Calibration required | No | Yes (small dataset) |
| Activations quantized | At runtime | Fixed (from calibration) |
| Inference speed | Faster than FP32, slower than static | Fastest — no runtime quantization overhead |
| Accuracy | Slightly lower (dynamic scale less precise) | Slightly higher (tuned scale) |
| Model change needed | None | Add QuantStub / DeQuantStub |
| Best for | Quick application, NLP, no calib data | CV models, max throughput, edge export |

---

## Part 5 — Quantization Caveats

### CPU only

PyTorch's standard quantization (`quantize_dynamic`, `prepare`/`convert`) runs on CPU. The `fbgemm` backend targets x86; `qnnpack` targets ARM (mobile). GPU quantization requires TensorRT or torch's experimental `torch.ao` path.

In the demo, the model is trained on GPU, then moved to CPU for all quantization steps and evaluation comparisons.

---

### Accuracy sensitivity

Not all models quantize cleanly. Accuracy degradation is larger when:

- Weights or activations have large outliers (scale is dominated by a few extreme values)
- The model is already near capacity (no redundancy to absorb rounding error)
- Quantization is applied layer by layer rather than end-to-end

Mitigation options: per-channel quantization (default in `fbgemm`), quantization-aware training (QAT), or mixed precision (keep sensitive layers in FP32).

---

### File size vs runtime size

The `int8.pth` file is smaller than `fp32.pth` because weights are stored as INT8. But `file_kb()` measures disk size, not runtime memory. At runtime, activations and intermediate buffers may still consume significant memory depending on the backend.

---

## Part 6 — Quantization vs Pruning vs Distillation

| | Pruning | Distillation | Quantization |
|---|---|---|---|
| Changes architecture | Optional (structured) | Yes (new model) | No |
| Changes precision | No | No | Yes (FP32→INT8) |
| Requires retraining | Fine-tuning after | Full student training | No (PTQ) / Optional (QAT) |
| Size reduction | Variable (sparsity) | Design-time | ~4× (INT8) |
| Inference speedup | Only with sparse kernels | Yes (smaller model) | Yes (INT8 matmul) |
| Stackable | Yes | Yes | Yes — last step |

The standard production pipeline applies all three:

```
1. Train large FP32 model
2. Knowledge distillation         → compact student architecture
3. Structured pruning on student  → fewer neurons / channels
4. INT8 quantization              → 4× size reduction, fast INT8 matmul
5. Export (ONNX / TensorRT)       → hardware-optimised deployment
```

---

## Quick Reference

### Dynamic quantization

```python
import torch.nn as nn

model_q = torch.quantization.quantize_dynamic(
    model, {nn.Linear}, dtype=torch.qint8
)
```

### Static PTQ

```python
# 1. Wrap model with QuantStub / DeQuantStub (see QuantizableMLP above)

# 2. Prepare
model.qconfig = torch.quantization.get_default_qconfig("fbgemm")  # x86
# model.qconfig = torch.quantization.get_default_qconfig("qnnpack")  # ARM
torch.quantization.prepare(model, inplace=True)

# 3. Calibrate
with torch.no_grad():
    for x, _ in calib_loader:
        model(x)

# 4. Convert
torch.quantization.convert(model, inplace=True)
```

### Check model size on disk

```python
import os
def file_kb(path):
    return os.path.getsize(path) / 1024

torch.save(model.state_dict(), "model.pth")
print(f"{file_kb('model.pth'):.1f} KB")
```

### Latency benchmark

```python
import time

def latency_ms(model, loader, n_batches=20):
    model.eval()
    t0 = time.perf_counter()
    with torch.no_grad():
        for i, (x, _) in enumerate(loader):
            model(x)
            if i >= n_batches:
                break
    return (time.perf_counter() - t0) / (n_batches + 1) * 1000
```

---

## Official Documentation

- PyTorch quantization overview: https://pytorch.org/docs/stable/quantization.html
- PTQ static tutorial: https://pytorch.org/tutorials/advanced/static_quantization_tutorial.html
- `quantize_dynamic`: https://pytorch.org/docs/stable/generated/torch.quantization.quantize_dynamic.html
- fbgemm (x86 backend): https://github.com/pytorch/FBGEMM
- ONNX quantization for export: https://onnxruntime.ai/docs/performance/model-optimizations/quantization.html
