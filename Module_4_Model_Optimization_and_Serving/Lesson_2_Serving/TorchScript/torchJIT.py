import torch
import torch.nn as nn
import torch.jit
import time


class SimpleModel(nn.Module):
    def __init__(self):
        super(SimpleModel, self).__init__()
        self.fc1 = nn.Linear(2, 10)
        self.fc2 = nn.Linear(10, 100)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.relu(self.fc1(x))
        x = self.fc2(x)
        return x


model = SimpleModel()
print("=== Original PyTorch Model ===")
print(model)

# torch.jit.trace records the operations executed during a forward pass with a
# concrete example input and compiles them into a static TorchScript graph.
# The result is a ScriptModule that can run without the Python interpreter,
# making it portable (e.g., deployable via LibTorch in C++).
traced_model = torch.jit.trace(model, torch.randn(1, 2))

print("\n=== Traced TorchScript Model ===")
print(traced_model)

# RecursiveScriptModule — the TorchScript wrapper around the original nn.Module
print("\nType:", type(traced_model))

# Human-readable TorchScript source generated from the traced graph
print("\n--- TorchScript Code ---")
print(traced_model.code)

# Low-level IR graph showing every tensor operation (useful for debugging)
print("\n--- Computation Graph ---")
print(traced_model.graph)

# --- Benchmark ---
# Warm up both models so lazy initialization doesn't skew the first timing
WARMUP = 50
RUNS = 5000
x = torch.randn(1, 2)

with torch.no_grad():
    for _ in range(WARMUP):
        model(x)
        traced_model(x)

    # Time the eager (standard Python) model
    start = time.perf_counter()
    for _ in range(RUNS):
        model(x)
    eager_time = time.perf_counter() - start

    # Time the TorchScript traced model (no Python overhead per call)
    start = time.perf_counter()
    for _ in range(RUNS):
        traced_model(x)
    jit_time = time.perf_counter() - start

print("\n=== Benchmark ({} runs) ===".format(RUNS))
print(f"  Eager model : {eager_time * 1000:.2f} ms total  |  {eager_time / RUNS * 1e6:.2f} µs/call")
print(f"  JIT model   : {jit_time  * 1000:.2f} ms total  |  {jit_time  / RUNS * 1e6:.2f} µs/call")
print(f"  Speedup     : {eager_time / jit_time:.2f}x")

# Sanity check — both models must produce identical outputs
out_eager = model(x)
out_jit   = traced_model(x)
print("\nOutputs match:", torch.allclose(out_eager, out_jit, atol=1e-5))


# =============================================================================
# PART 2 — The Silent Bug: where torch.jit.trace breaks
# =============================================================================
#
# torch.jit.trace works by recording a single execution path through the model.
# Any Python control flow that branches on a tensor VALUE (if/else, loops that
# depend on tensor content) is NOT compiled — only the path taken at trace time
# is frozen into the graph.  The other branches silently disappear.

class BranchModel(nn.Module):
    """Returns x*2 when the input sum is positive, x*(-1) otherwise."""

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.sum() > 0:   # branch decision depends on a tensor value at runtime
            return x * 2
        else:
            return x * -1


branch_model = BranchModel()

# Trace with a POSITIVE input — the tracer records only the `x * 2` branch.
pos_input = torch.tensor([[1.0, 2.0]])
traced_branch = torch.jit.trace(branch_model, pos_input)

print("\n\n=== PART 2: trace silently ignores control flow ===")
print("Traced TorchScript code (notice: only ONE branch was captured):")
print(traced_branch.code)

# Now run the traced model with a NEGATIVE input — it should take `x * -1`,
# but instead it silently returns `x * 2` because the else-branch was never
# recorded.  No exception is raised; the result is just wrong.
neg_input = torch.tensor([[-1.0, -2.0]])

out_eager_neg  = branch_model(neg_input)         # correct: x * -1
out_traced_neg = traced_branch(neg_input)         # WRONG  : still x * 2

print(f"\nEager  output for negative input : {out_eager_neg.tolist()}")
print(f"Traced output for negative input : {out_traced_neg.tolist()}  <-- SILENT BUG")
print("Results match? ", torch.allclose(out_eager_neg, out_traced_neg))


# =============================================================================
# PART 3 — The Fix: torch.jit.script
# =============================================================================
#
# torch.jit.script compiles the Python *source code* of the module rather than
# recording a single execution trace.  It understands TorchScript's subset of
# Python (type annotations required) and correctly captures all branches,
# loops, and recursion — making it the right tool whenever logic depends on
# tensor values at runtime.

scripted_branch = torch.jit.script(BranchModel())

print("\n\n=== PART 3: torch.jit.script captures all branches ===")
print("Scripted TorchScript code (both branches present):")
print(scripted_branch.code)

out_scripted_pos = scripted_branch(pos_input)
out_scripted_neg = scripted_branch(neg_input)

print(f"\nScripted output for positive input : {out_scripted_pos.tolist()}  (expected {(pos_input * 2).tolist()})")
print(f"Scripted output for negative input : {out_scripted_neg.tolist()}  (expected {(neg_input * -1).tolist()})")
print("Positive correct?", torch.allclose(out_scripted_pos, pos_input * 2))
print("Negative correct?", torch.allclose(out_scripted_neg, neg_input * -1))

# --- Benchmark: eager vs traced vs scripted on BranchModel ---
with torch.no_grad():
    for _ in range(WARMUP):
        branch_model(pos_input)
        traced_branch(pos_input)
        scripted_branch(pos_input)

    start = time.perf_counter()
    for _ in range(RUNS):
        branch_model(pos_input)
    t_eager = time.perf_counter() - start

    start = time.perf_counter()
    for _ in range(RUNS):
        traced_branch(pos_input)
    t_traced = time.perf_counter() - start

    start = time.perf_counter()
    for _ in range(RUNS):
        scripted_branch(pos_input)
    t_scripted = time.perf_counter() - start

print(f"\n=== BranchModel Benchmark ({RUNS} runs) ===")
print(f"  Eager    : {t_eager    * 1000:.2f} ms  |  {t_eager    / RUNS * 1e6:.2f} µs/call")
print(f"  Traced   : {t_traced   * 1000:.2f} ms  |  {t_traced   / RUNS * 1e6:.2f} µs/call  (speedup {t_eager/t_traced:.2f}x)")
print(f"  Scripted : {t_scripted * 1000:.2f} ms  |  {t_scripted / RUNS * 1e6:.2f} µs/call  (speedup {t_eager/t_scripted:.2f}x)")
