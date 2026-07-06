# Lesson 4.1 — Model Compression

| | |
|---|---|
| **Problem this solves** | A model that achieves great accuracy during training may be too large to fit on an edge device, too slow for real-time inference, or too expensive to serve at scale. Compression closes the gap between training accuracy and deployment reality. |
| **Mental model** | Four techniques in order of invasiveness: pruning removes redundant weights; quantization reduces numerical precision; distillation trains a smaller model to replicate a larger one; ONNX decouples the model from the training framework. Apply them in sequence until deployment constraints are met. |
| **What the lecture demonstrates** | Four sub-demos — each a standalone technique applied to MNIST classification — showing the size/speed/accuracy tradeoff at each compression stage |
| **Where this fits** | This is the **Optimization Layer** in the system map. The compressed model feeds the Serving Layer (Lesson 1.1) and is the last step before deployment. |

---

## Structure

```
Lesson_1_Compression/
  COMPRESSION_OVERVIEW.md   ← start here: the full compression pipeline
  pruning/
    PRUNING_GUIDE.md + structured_prune.ipynb + unstructured_prune.ipynb
  Quantization/
    QNT_GUIDE.md + qnt.py
  KD/
    KD_GUIDE.md + kd.py
  onnx/
    ONNX_GUIDE.md + onnx_export.py
```

**Read order:** `COMPRESSION_OVERVIEW.md` first to understand the full pipeline,
then follow into whichever sub-demo you're working through.

---

## The Compression Sequence

```
Original model (FP32, full weights)
      │
      ▼  pruning/       — remove near-zero weights (unstructured or structured)
Sparse model
      │
      ▼  Quantization/  — FP32 → INT8 (dynamic or static PTQ)
Quantized model
      │
      ▼  KD/            — optional: train a smaller student model on teacher outputs
Student model
      │
      ▼  onnx/          — export to hardware-agnostic ONNX format
ONNX model  →  deploy anywhere
```
