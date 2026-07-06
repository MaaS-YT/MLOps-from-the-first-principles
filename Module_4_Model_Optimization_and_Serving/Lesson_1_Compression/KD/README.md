# Lesson 4.1c — Knowledge Distillation

| | |
|---|---|
| **Problem this solves** | Sometimes pruning and quantization aren't enough — the architecture itself is too large. Knowledge distillation trains a small "student" model to mimic a large "teacher", transferring capability without transferring size. |
| **Mental model** | Hard labels (0 or 1) carry minimal information. The teacher's soft output distribution — e.g., 0.7 cat, 0.2 dog, 0.1 bird — carries rich structure about what the model has learned. Training the student on these soft targets (via KL divergence) transfers that structure. Temperature scaling controls how "soft" the distribution is. |
| **What the demo shows** | Teacher CNN → Student CNN on MNIST — KL divergence loss, temperature parameter, accuracy vs size tradeoff |
| **Where this fits** | Optional third step in the compression pipeline. Use when the architecture itself needs to shrink, not just the weights. |

---

## Files

| File | Purpose |
|------|---------|
| `KD_GUIDE.md` | Full guide: soft labels, KL divergence, temperature scaling, T² compensation |
| `kd.py` | Teacher training → student distillation with annotated loss function |

**Start with:** `KD_GUIDE.md`

```bash
uv run --no-sync python kd.py
```
