# Module 4 — Model Optimization and Serving

*The model that trains is rarely the model that deploys.*

A model that achieves great accuracy on a GPU during training may be too large, too slow,
or too framework-specific to run in the target deployment environment.
This module closes the gap between training and deployment — through compression,
format standardization, and high-throughput serving protocols.

---

## Lessons

| Lesson | Topic | Problem It Solves | Guide | Status |
|--------|-------|-------------------|-------|--------|
| 4.1 | [Compression](Lesson_1_Compression/) | The model is too large or too slow for the deployment target | [COMPRESSION_OVERVIEW.md](Lesson_1_Compression/COMPRESSION_OVERVIEW.md) | ✓ |
| 4.2 | [Serving](Lesson_2_Serving/) | The model is coupled to Python and can't reach C++ environments or high-throughput service interfaces | [TorchScript](Lesson_2_Serving/TorchScript/TORCHSCRIPT_GUIDE.md) · [LibTorch](Lesson_2_Serving/LibTorch/LIBTORCH_GUIDE.md) · [gRPC](Lesson_2_Serving/API_gRPC/GRPC_GUIDE.md) | ✓ |

---

## What This Module Builds

```
OPTIMIZATION LAYER (Lesson 4.1)
  Pruning       →  remove near-zero weights
  Quantization  →  FP32 weights → INT8 (4× smaller, 2–4× faster)
  Distillation  →  train a small model to mimic a large one
  ONNX export   →  hardware-agnostic format for any runtime

SERVING LAYER — high throughput (Lesson 4.2, coming)
  gRPC          →  binary protocol, lower latency than REST, streaming support

SERIALIZATION (Lesson 4.3, coming)
  TorchScript / Protobuf  →  format tradeoffs for edge and cloud targets
```

---

## Where This Fits

This module sits between the Reproducibility layer (Module 2) and Production Engineering
(Module 5). The training process (Module 2) produces an accurate model. This module
transforms it into a model that can actually be deployed: smaller, faster, and
independent of the training framework.

Open `SYSTEM_MAP.md` at the repo root for the full system view.
