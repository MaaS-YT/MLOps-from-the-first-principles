# Lesson 4.2a — TorchScript

| | |
|---|---|
| **Problem this solves** | Eager PyTorch models are Python objects — they can't run without a Python interpreter, they can't be deployed to C++, and every forward call pays Python's per-operation dispatch overhead. |
| **Mental model** | TorchScript is a statically-typed, compilable subset of Python. `torch.jit.trace` records one execution path; `torch.jit.script` compiles the full source. The compiled module is portable: it runs in C++, mobile, and anywhere LibTorch exists — no Python required. |
| **What the demo shows** | `trace` on a simple MLP → benchmark eager vs JIT → `trace` silently breaking on data-dependent control flow → `torch.jit.script` correctly capturing both branches → three-way benchmark |
| **Where this fits** | First step in the serving pipeline. TorchScript is the compilation stage; LibTorch (Lesson 4.2b) is the C++ runtime stage. |

---

## Files

| File | Purpose |
|------|---------|
| `TORCHSCRIPT_GUIDE.md` | Full guide: trace vs script, when each applies, the silent-bug failure mode, benchmark interpretation |
| `torchJIT.py` | Part 1: trace a simple MLP → Part 2: trace silent bug demo → Part 3: script fix + benchmark |

**Start with:** `TORCHSCRIPT_GUIDE.md`

```bash
uv run TorchScript/torchJIT.py
```
