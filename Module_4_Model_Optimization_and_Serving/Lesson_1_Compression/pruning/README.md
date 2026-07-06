# Lesson 4.1a — Pruning

| | |
|---|---|
| **Problem this solves** | Neural networks are over-parameterized by design — many weights contribute almost nothing to the output. Pruning removes them, reducing model size and (with structured pruning) enabling real hardware speedups. |
| **Mental model** | Two variants: unstructured pruning zeroes individual weights anywhere in the matrix (high sparsity ratio, no speedup without sparse hardware); structured pruning removes entire filters or neurons (lower sparsity ratio, real speedup on any hardware). |
| **What the demo shows** | Both variants applied to an MNIST CNN — sparsity ratio, size reduction, and accuracy retention at each step |
| **Where this fits** | First step in the compression pipeline. Feeds into quantization. |

---

## Files

| File | Purpose |
|------|---------|
| `PRUNING_GUIDE.md` | Full guide: unstructured vs structured, `torch.nn.utils.prune`, hardware implications |
| `unstructured_prune.ipynb` | Weight-level pruning with L1 magnitude criterion |
| `structured_prune.ipynb` | Filter-level pruning — removes entire output channels |

**Start with:** `PRUNING_GUIDE.md`

```bash
uv run --no-sync jupyter notebook unstructured_prune.ipynb
uv run --no-sync jupyter notebook structured_prune.ipynb
```
