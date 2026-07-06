# Lesson 4.2a — TorchScript

> **Lesson 4.2 / Part 1** — Compile a PyTorch model from Python into a portable, interpreter-free representation. The key insight: TorchScript captures the computation graph, not the Python code — enabling deployment anywhere LibTorch runs.

| | |
|---|---|
| **Problem this solves** | Eager PyTorch is tightly coupled to Python. Every `model(x)` call dispatches through the Python interpreter, dispatches into C++ kernels, and returns Python objects. This works fine for training. In production — a C++ inference server, a mobile app, a real-time system — Python is unavailable or unacceptably slow. |
| **Mental model** | TorchScript is a compiled, statically-typed subset of Python. `torch.jit.trace` and `torch.jit.script` both produce a `ScriptModule` — a self-contained object that carries the computation graph and weights, runnable without Python. |
| **What the demo covers** | `torch.jit.trace` on a simple MLP → benchmark → the silent control-flow bug → `torch.jit.script` as the fix. |

---

## Part 1 — Why TorchScript

### The eager mode problem

PyTorch's default execution mode is **eager**: every operation executes immediately as Python calls it. This is great for development — you get Python's full debuggability, dynamic typing, and control flow. But it has a deployment cost:

- The Python interpreter must be present on every machine that runs inference.
- Every forward pass dispatches through Python before reaching C++ kernels.
- The graph is re-traced on each call; no ahead-of-time optimisation is possible.

```
Eager forward pass:
  Python code → Python interpreter → C++ dispatch → kernel → Python return
                 ^^^^^^^^^^^^^^^^^^^
                 overhead per op, can't be removed, can't be optimized across ops
```

TorchScript eliminates the middle step by compiling the model once.

---

### What TorchScript actually is

TorchScript is a **statically-typed subset of Python** designed to be:

1. **Compilable** — the compiler can analyse types and build an IR graph at export time
2. **Serialisable** — the resulting IR + weights fit in a single `.pt` file
3. **Portable** — the `.pt` file runs on any machine with LibTorch, including C++ and mobile

```
TorchScript compilation:
  Python module → IR graph (operations as nodes, tensors as edges)
                  ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
                  all operations fused, Python overhead eliminated,
                  can be loaded by C++ torch::jit::load()
```

---

### Where TorchScript fits in the serving pipeline

```
1. Train to convergence     Python, eager, full framework
2. Compress                 pruning / quantization / distillation  (Lesson 4.1)
3. Compile  ← you are here  torch.jit.script → model.pt
4. Deploy                   C++ LibTorch runtime  (Lesson 4.2b)
```

---

## Part 2 — trace vs script

Two compilation paths. They produce the same output type (`ScriptModule`) but work differently.

### torch.jit.trace

`trace` runs the model once with a concrete example input and **records every tensor operation** that executes. The sequence of recorded operations becomes the graph.

```python
traced = torch.jit.trace(model, torch.randn(1, 2))
```

**What it captures:** every op executed during that single forward pass.
**What it misses:** any Python control flow that branches on a tensor value — only the branch taken during tracing is recorded.

```
Trace execution:
  model(example_input) → records ops A → B → D (branch taken)
                          silently discards ops C (branch not taken)
                          frozen graph: always runs A → B → D
```

---

### torch.jit.script

`script` **compiles the Python source code** of the module. It analyses all branches, loops, and recursive calls — without running them first.

```python
scripted = torch.jit.script(model)
```

**What it captures:** the full source — all branches, all paths.
**Requirements:** TorchScript-compatible code and type annotations on `forward()`.

```
Script compilation:
  source code → type checker → TorchScript IR
                                ↓
                        if x.sum() > 0:
                            return x * 2   ← both branches in graph
                        else:
                            return x * -1
```

---

### Choosing between them

| | trace | script |
|---|---|---|
| Control flow on tensor values | Silently breaks | Handles correctly |
| Python lists / dicts / strings | Works | Works |
| Loops with dynamic length | Only unrolls the traced length | Handles correctly |
| Type annotations required | No | Yes |
| Works on third-party modules | Yes | May fail if code is not TorchScript-compatible |
| **Use when** | No data-dependent branches, simple feed-forward model | Anything with `if`/`for`/recursion on tensor content |

**Default recommendation:** use `torch.jit.script` for production exports. It catches problems at export time, not at deployment time.

---

## Part 3 — The Silent Bug

This is the most important failure mode to understand. `torch.jit.trace` does **not raise an error** when it misses a branch — it silently bakes in the wrong behaviour.

### Example

```python
class BranchModel(nn.Module):
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.sum() > 0:       # branch depends on tensor VALUE at runtime
            return x * 2
        else:
            return x * -1
```

Tracing with a positive input:

```python
traced = torch.jit.trace(model, torch.tensor([[1.0, 2.0]]))  # positive → takes x*2 branch
```

Inspecting the compiled code:

```python
print(traced.code)
# def forward(self, x: Tensor) -> Tensor:
#   return torch.mul(x, CONSTANTS.c0)    ← x*2 baked in as a constant
#                                           the else branch is gone
```

Running with a negative input:

```python
traced(torch.tensor([[-1.0, -2.0]]))    # should return [1.0, 2.0]  (x * -1)
# returns: [[-2.0, -4.0]]               # WRONG — still applies x * 2
```

PyTorch does fire a `TracerWarning` here, but warnings are easy to miss and the model runs — silently wrong.

---

### Why trace can't detect this

The trace is a recording of one execution, not a proof of correctness. The `if x.sum() > 0` check converts a tensor to a Python bool — at that moment, the value is concrete and the branch is taken. There is nothing in the trace to indicate the branch exists; it's just gone from the graph.

---

### The script fix

```python
scripted = torch.jit.script(BranchModel())

print(scripted.code)
# def forward(self, x: Tensor) -> Tensor:
#   if bool(torch.gt(torch.sum(x), 0)):
#     _0 = torch.mul(x, 2)             ← both branches present
#   else:
#     _0 = torch.mul(x, -1)
#   return _0
```

Both branches are in the compiled IR. The model is now correct for all inputs.

---

## Part 4 — Benchmark Interpretation

The demo benchmarks eager vs traced vs scripted on `BranchModel` (5 000 runs, CPU).

### What the numbers mean

TorchScript's speedup on small models running on CPU is often **near 1× or slightly below**. This is expected and not a problem.

```
Why trace/script can be slower than eager on small CPU models:

  Eager model:   Python → C++ dispatch → tiny kernel (e.g., mul) → Python return
  JIT model:     JIT interpreter dispatch → tiny kernel → return

  The kernel (mul) runs in microseconds.
  The dispatch overhead is comparable to the kernel time.
  JIT dispatch is not always cheaper than Python dispatch for trivial ops.
```

TorchScript's real advantages appear when:

- **Model is large** — the ratio of dispatch overhead to compute time shrinks
- **Running on GPU** — JIT can fuse kernel launches, reducing GPU round-trips
- **Deployed without Python** — this is the primary reason for TorchScript. Zero overhead vs Python because there is no Python at all.
- **Mobile / embedded** — Python is unavailable; LibTorch is the only option

The benchmark in this demo demonstrates the compilation is correct, not that it is faster on toy CPU models.

---

## Part 5 — Reading the Compiled Graph

Two inspection tools are available on any `ScriptModule`:

### `.code` — human-readable TorchScript source

```python
print(scripted.code)
```

Shows the model as TorchScript Python — the compiled representation, not the original source. Useful for verifying that all branches are present and ops are as expected.

### `.graph` — low-level IR

```python
print(scripted.graph)
```

Shows the internal IR nodes — each tensor operation as a typed node with input/output edges. Useful for debugging fusions, verifying constant folding, or understanding what the backend actually executes.

---

## Quick Reference

### Trace a model

```python
traced = torch.jit.trace(model, example_input)
traced.save("model.pt")
```

### Script a model

```python
scripted = torch.jit.script(model)   # requires type annotations on forward()
scripted.save("model.pt")
```

### Inspect the compiled graph

```python
print(scripted.code)    # human-readable
print(scripted.graph)   # IR nodes
```

### Load a saved model

```python
loaded = torch.jit.load("model.pt")
output = loaded(input_tensor)
```

---

## Official Documentation

- TorchScript introduction: https://pytorch.org/docs/stable/jit.html
- torch.jit.trace: https://pytorch.org/docs/stable/generated/torch.jit.trace.html
- torch.jit.script: https://pytorch.org/docs/stable/generated/torch.jit.script.html
- TorchScript language reference: https://pytorch.org/docs/stable/jit_language_reference.html
