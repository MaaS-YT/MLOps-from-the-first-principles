# Lesson 4.1b — Quantization

| | |
|---|---|
| **Problem this solves** | FP32 weights are four bytes each. INT8 weights are one byte. Quantization cuts model size by up to 4× and inference time by 2–4× on modern CPUs — with minimal accuracy loss on most tasks. |
| **Mental model** | Two approaches: dynamic quantization replaces weight matrices at inference time (easy, weights only); static PTQ uses calibration data to quantize both weights and activations (more effort, better speedup). Both require `QuantStub`/`DeQuantStub` to mark graph boundaries. |
| **What the demo shows** | Dynamic INT8 and static PTQ applied to an MNIST CNN — size before/after, inference time before/after, accuracy delta |
| **Where this fits** | Second step in the compression pipeline. Follows pruning, feeds into KD or ONNX export. |

---

## Files

| File | Purpose |
|------|---------|
| `QNT_GUIDE.md` | Full guide: dynamic vs static quantization, observer calibration, fbgemm vs qnnpack |
| `qnt.py` | Both quantization paths with before/after benchmarks |

**Start with:** `QNT_GUIDE.md`

```bash
uv run --no-sync python qnt.py
```
