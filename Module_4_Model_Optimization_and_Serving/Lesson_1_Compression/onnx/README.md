# Lesson 4.1d — ONNX Export

| | |
|---|---|
| **Problem this solves** | A PyTorch model only runs in Python with PyTorch installed. ONNX breaks that coupling — export once, run anywhere: ONNX Runtime, TensorRT, CoreML, TFLite, browser, edge device. |
| **Mental model** | ONNX is a standardized computation graph format. PyTorch traces the model with a dummy input, records every operation as a graph node, and serializes it to a `.onnx` file. The training framework is no longer needed at inference time. |
| **What the demo shows** | PyTorch MNIST model → ONNX export → ONNX Runtime inference — latency comparison, output equivalence check |
| **Where this fits** | Final step in the compression pipeline. The ONNX model is what gets deployed to the serving layer. |

---

## Files

| File | Purpose |
|------|---------|
| `ONNX_GUIDE.md` | Full guide: ONNX graph format, export mechanics, opset versions, runtime providers |
| `onnx_export.py` | Export + validate + benchmark against native PyTorch |

**Start with:** `ONNX_GUIDE.md`

```bash
uv run --no-sync python onnx_export.py
```
