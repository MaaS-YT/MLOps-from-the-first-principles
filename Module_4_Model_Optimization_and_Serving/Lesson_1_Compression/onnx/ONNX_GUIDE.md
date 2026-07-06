# Lesson 4.1d — ONNX Export

> **Lesson 4.1 / Sub-demo 4** — After compression, export to a hardware-agnostic format that any inference runtime can consume. ONNX is the standard interchange layer between training frameworks and deployment targets.

| | |
|---|---|
| **Problem this solves** | A PyTorch model is coupled to PyTorch's runtime. Deploying on edge hardware, mobile, or a high-performance inference server often requires a different runtime. ONNX decouples the model from the framework. |
| **Mental model** | ONNX is a serialized computation graph — not Python, not PyTorch. Any runtime that understands the ONNX spec can load and execute it: ONNX Runtime, TensorRT, CoreML, TFLite converters. One export, many targets. |
| **What the lecture demonstrates** | Export an MLP to ONNX → inspect the protobuf graph structure → run inference with ONNX Runtime → benchmark all four ORT optimization levels. |
| **Where this fits** | Final step in the compression stack — the output format that enters the deployment pipeline. |

---

# ONNX — Under the Hood

This guide does not stop at "ONNX is an interchange format." It explains
what the bytes on disk actually are, how PyTorch converts your model into
those bytes, and what ONNX Runtime does with them before the first
multiplication runs.

---

## What Problem ONNX Solves

Training a neural network and deploying it are done by completely different
software stacks. PyTorch is the dominant training framework but is not a
production inference runtime — it carries millions of lines of Python,
autograd machinery, optimizer state, and training-only ops that serve no
purpose in inference. A production system needs:

- A binary that loads in under 10ms with no Python interpreter
- Support for heterogeneous hardware (x86 CPU, ARM, NVIDIA GPU, Apple Silicon)
- Deterministic latency (no JIT recompilation surprises)
- A model file that is framework-agnostic (the serving team may not use Python)

ONNX addresses this by defining a **common serialization format** that any
training framework can write to and any inference runtime can read from.
The model becomes a self-contained binary. PyTorch, TensorFlow, and JAX all
export to ONNX. ONNX Runtime, TensorRT, CoreML, and OpenVINO all import it.

---

## The Binary Format: Protocol Buffers

Every `.onnx` file is a Protocol Buffer (protobuf) binary. Protobuf is a
binary serialization format developed at Google — it is compact, fast to
parse, and schema-driven. The schema that defines ONNX is in
[`onnx/onnx.proto3`](https://github.com/onnx/onnx/blob/main/onnx/onnx.proto3)
in the ONNX repository.

You can verify this yourself: open any `.onnx` file in a hex editor. The
first bytes are protobuf field tags. There is no magic number, no JSON
header, no compression — just raw protobuf.

The `onnx` Python package auto-generates `onnx_ml_pb2.py` from this schema.
When you call `onnx.load("model.onnx")`, you get back a Python object whose
type is `ModelProto` — the protobuf message class for the top-level ONNX
container.

---

## The Protobuf Hierarchy

```
ModelProto                        ← the entire .onnx file
│
├── ir_version: int64             ← ONNX IR format version (10 as of ONNX 1.16+)
├── producer_name: string         ← "pytorch"
├── producer_version: string      ← "2.12.0"
│
├── opset_import[]                ← which opset(s) this model uses
│     └── OperatorSetIdProto
│           ├── domain: string    ← "" = standard ONNX ops
│           │                       "com.microsoft" = ORT extensions
│           │                       "org.pytorch.aten" = ATen fallback ops
│           └── version: int64    ← opset number (e.g., 18)
│
└── graph: GraphProto             ← the computation graph
      │
      ├── name: string
      │
      ├── node[]                  ← OPERATIONS (Conv, Gemm, Relu, ...)
      │     └── NodeProto
      │           ├── op_type: string   "Gemm", "Relu", "Reshape"
      │           ├── domain: string    "" for standard ops
      │           ├── input[]: string   tensor names this op reads
      │           ├── output[]: string  tensor names this op writes
      │           └── attribute[]       op-specific config
      │                 └── AttributeProto
      │                       ├── name: string
      │                       ├── type: enum (FLOAT, INT, STRING, FLOATS, INTS, ...)
      │                       └── value field (f, i, s, floats, ints, ...)
      │
      ├── input[]                 ← model-level inputs (with shape + dtype)
      │     └── ValueInfoProto
      │           ├── name: string
      │           └── type: TypeProto
      │                 └── tensor_type: Tensor_TypeProto
      │                       ├── elem_type: int  (1=FLOAT, 3=INT8, 7=INT64)
      │                       └── shape: TensorShapeProto
      │                             └── dim[]
      │                                   ├── dim_value: int64  (fixed dimension)
      │                                   └── dim_param: string (symbolic: "batch_size")
      │
      ├── output[]                ← model-level outputs (same structure)
      │
      ├── initializer[]           ← WEIGHTS (the actual float arrays)
      │     └── TensorProto
      │           ├── name: string   matches an input[] entry of a NodeProto
      │           ├── data_type: int (1=FLOAT, 3=INT8, ...)
      │           ├── dims[]: int64  shape, e.g. [256, 784]
      │           └── raw_data: bytes  row-major float32 bytes
      │                                (this is the bulk of the file size)
      │
      └── value_info[]            ← intermediate tensor shapes
            └── ValueInfoProto    (empty until shape_inference.infer_shapes runs)
```

The key insight is that the graph is a **data-flow DAG encoded entirely as
strings**: a NodeProto reads tensors whose names appear in its `input[]` list
and writes tensors whose names appear in its `output[]` list. The graph has
no pointers, no references — just strings. The runtime resolves names to
actual memory buffers at session creation time.

---

## How torch.onnx.export Works

### Step 1: JIT Tracing

`torch.onnx.export` calls `torch.jit.trace` on your model with the dummy
input. Tracing works by executing the model's forward pass once in a
modified dispatch mode where every tensor operation records itself in a
JIT IR graph rather than immediately computing a result.

The JIT IR is an SSA (Static Single Assignment) intermediate representation.
Every op produces a named value; no value is assigned twice. This maps
directly to ONNX's string-name data-flow model.

```
Python:     x = F.relu(self.fc1(x))

JIT IR:     %linear_out = aten::linear(%x, %fc1_weight, %fc1_bias)
            %relu_out   = aten::relu(%linear_out)
```

**What tracing misses**: the tracer follows the single execution path taken
during the trace run. If your forward method contains:

```python
if x.shape[0] > 1:   # data-dependent branch
    ...
```

Only one branch is recorded. The other branch is silently dropped. Similarly,
Python loops whose iteration count depends on a tensor value are unrolled to
however many iterations the dummy input causes. This is not a bug — it is an
inherent limitation of execution-based tracing.

For models with true dynamic control flow, use `torch.onnx.dynamo_export`
(PyTorch ≥ 2.0), which uses TorchDynamo to capture the symbolic graph
without executing it.

### Step 2: Symbolic Op Mapping

Once the JIT IR is complete, PyTorch walks every node and calls the
corresponding ONNX **symbolic function** for that op. These functions live in
`torch/onnx/symbolic_opset*.py` in the PyTorch source tree.

Each symbolic function receives the JIT node and returns ONNX `NodeProto`
objects. For example, `aten::linear` maps to `onnx::Gemm`:

```python
# simplified from torch/onnx/symbolic_opset9.py
def linear(g, input, weight, bias):
    output = g.op("Gemm", input, weight, bias,
                  alpha_f=1.0, beta_f=1.0, transB_i=1)
    return output
```

The `transB_i=1` attribute tells Gemm to transpose the weight matrix —
because PyTorch stores Linear weights as `[out_features, in_features]` but
ONNX Gemm expects `A @ B` where B can optionally be transposed.

### Step 3: Initializer Collection

Every `nn.Parameter` that was accessed during the trace is collected as a
`TensorProto` initializer. The parameter's `data` tensor (a numpy array) is
serialized to `raw_data` bytes.

For our MLP this produces 7 initializers: fc1.weight, fc1.bias, fc2.weight,
fc2.bias, fc3.weight, fc3.bias, and `val_3` (a constant `[-1]` tensor used
in the Reshape op for flattening).

### Step 4: Serialization

The completed `GraphProto` is wrapped in a `ModelProto` with metadata, then
serialized to bytes via protobuf and written to disk. For a model with 235K
parameters at float32, the file is:

```
235,147 params × 4 bytes/param = 940 KB raw weights
+ node/attribute overhead        ≈ 926 KB total
```

---

## The Opset System

The opset version controls which operator *definitions* are valid. Each
ONNX opset release either adds new operators or changes the semantics of
existing ones (new attributes, relaxed type constraints, different shape
propagation rules).

For example:

| Op      | Added in opset | Notable change |
|---------|---------------|----------------|
| Relu    | 1             | — |
| Gemm    | 1             | transA/transB added in opset 6 |
| Reshape | 1             | shape as input (not attribute) from opset 5 |
| Resize  | 10            | replaces Upsample |
| GroupNormalization | 18 | new in 18 |

When you pass `opset_version=18` to `torch.onnx.export`, PyTorch uses the
symbolic functions from `symbolic_opset18.py` (and falls back to earlier
opsets for ops that didn't change). The resulting model file contains
`opset_import { version: 18 }` and every NodeProto uses op definitions
from that opset.

ONNX Runtime supports opsets 7 through 22 (as of ORT 1.18+). Always use
the highest opset your target runtime supports — older opsets have
restrictions (e.g., opset 7 requires static shapes everywhere).

---

## Shape Inference

After export, the `value_info` list in the graph is empty. All intermediate
tensors exist only as string names with no recorded shapes. This is fine for
inference but makes tools like Netron show unknown dimensions, and some
ORT optimization passes require known shapes.

`onnx.shape_inference.infer_shapes(model)` runs shape propagation:

1. It starts from the model inputs (whose shapes are known: `[batch_size, 1, 28, 28]`).
2. For each node, it calls the registered `shape_inference_function` for that op.
3. Each function computes the output shape from the input shapes symbolically.
4. The result is stored as a new `ValueInfoProto` entry in `value_info`.

For example, Shape inference for our graph:

```
input:   [batch_size, 1, 28, 28]
  │
  ▼ Shape(start=0, end=1) → [1]    (extracts batch dimension)
  ▼ Concat with [-1]      → [2]    (builds reshape target: [batch_size, -1])
  ▼ Reshape               → [batch_size, 784]
  ▼ Gemm(fc1)             → [batch_size, 256]
  ▼ Relu                  → [batch_size, 256]
  ▼ Gemm(fc2)             → [batch_size, 128]
  ▼ Relu                  → [batch_size, 128]
  ▼ Gemm(fc3)             → [batch_size, 10]
logits:  [batch_size, 10]
```

The `batch_size` string is a **symbolic dimension** — it propagates through
as a string name wherever it appears, allowing the model to accept any batch
size at runtime. Fixed integer dimensions propagate as integers.

This is entirely symbolic computation — no actual data moves through the
graph during shape inference.

---

## ONNX Runtime: The Execution Pipeline

When you call `ort.InferenceSession(onnx_path, ...)`, ORT executes a
four-phase pipeline before any user data flows through.

### Phase 1: Model Loading and Validation

ORT deserializes the `ModelProto` from disk and checks:
- All `op_type` values have a registered kernel in the kernel catalogue
- Input/output names are consistent (no dangling references)
- Data types are compatible with operator constraints
- Opset version is supported

If any check fails, the constructor raises immediately with a descriptive
error — before any inference attempt.

### Phase 2: Graph Optimization

ORT runs a sequence of **graph rewrite passes** over the graph. The level
is controlled by `SessionOptions.graph_optimization_level`:

#### BASIC (1) — Structural simplification

These passes are purely structural and cannot change numerical results:

**Constant folding**: Any subgraph whose inputs are all constants is
evaluated at session-creation time and replaced with a single `Constant`
node. In our MLP the `Shape → Concat → val_3` subgraph (which computes the
reshape target `[batch_size, -1]`) has `val_3 = [-1]` as a constant
initializer, so ORT can pre-evaluate part of this chain.

**Identity elimination**: `Identity(x) → x`. Exported models often contain
these because the PyTorch exporter inserts them for type compatibility.

**Common subexpression elimination (CSE)**: If two nodes compute identical
outputs from identical inputs, one is removed and the other's output is
shared. This is rare in feedforward networks but common in Transformers
where the same query is projected multiple times.

**Dead code elimination**: Nodes whose outputs are not reachable from any
graph output are removed.

#### EXTENDED (2) — Kernel fusion

These passes merge multiple nodes into single composite kernels to eliminate
memory round-trips between operations:

**MatMul + Add → Gemm**: A `MatMul` node followed by an `Add` for bias is
collapsed into a single `Gemm` node, which maps to a single BLAS call
(e.g., `cblas_sgemm` on CPU). Our model already exports as Gemm directly,
so this pass finds nothing to do.

**Conv + BatchNormalization → ConvBatchNorm** (the most important for CNNs):
During training, BatchNorm maintains running mean/variance and has learnable
scale/shift. After training these are constants. The fusion absorbs the BN
parameters directly into the Conv weight and bias:

```
merged_weight = γ / sqrt(σ² + ε) × W
merged_bias   = β - γ × μ / sqrt(σ² + ε) + (γ / sqrt(σ² + ε)) × b
```

The resulting single Conv does exactly the same arithmetic in one kernel
call instead of two. For a ResNet-50, this eliminates 49 BatchNorm nodes
entirely.

**Conv + Relu → ConvRelu**: The activation is fused into the convolution
kernel. In AVX2 implementations this means the result of the multiply-
accumulate loop is clamped before being written to output memory, saving
one full read-write pass over the output tensor.

**Gelu approximation and layer norm fusions** (relevant for Transformers):
Patterns of `Mul + Add + Tanh + ...` that implement Gelu are replaced with
a single `Gelu` node backed by an optimized kernel.

#### ALL (3) — Layout optimization

The heaviest pass: ORT rewrites the physical memory layout of weight tensors
to match the preferred access pattern of the target hardware.

**NCHWc for CPU (x86 AVX2/AVX512)**: For convolutions, the channel
dimension is blocked — e.g., NCHW becomes NCHW8c (8 channels packed
contiguously). This allows the inner conv loop to load a contiguous block
of input channels into an AVX2 register (256-bit = 8 × float32) and process
all 8 channels in parallel with a single `vfmadd` instruction.

**NHWC for CUDA**: cuDNN's fastest convolutional kernels prefer NHWC layout.
ORT inserts layout-conversion nodes at the boundaries and rewrites all
internal Conv/BN/Relu nodes to NHWC.

These layout transforms are transparent to the user — the model's named
inputs and outputs stay in their original layout.

### Phase 3: Execution Provider Partitioning

ORT walks the optimized graph and assigns each node to an **Execution
Provider** (EP). It queries each EP's registered kernel catalogue in priority
order. If the EP has a kernel for this op, the node is assigned to it.
If not, ORT tries the next EP, ultimately falling back to CPU.

For nodes at EP boundaries (a CUDA node consuming the output of a CPU node),
ORT automatically inserts `MemcpyToDevice` / `MemcpyFromHost` nodes to
handle data movement.

### Phase 4: Execution Plan

ORT generates a linear schedule of kernel calls. It also builds a memory plan:
intermediate activation tensors are pre-allocated (or memory is reused across
non-overlapping lifetimes). This static allocation means inference has near-
zero dynamic memory allocation overhead — unlike PyTorch's eager mode, which
calls `cudaMalloc` for every intermediate tensor.

---

## Execution Providers In Depth

### CPUExecutionProvider

Uses **MLAS** (Microsoft Linear Algebra Subprograms), a custom SIMD kernel
library tuned for x86-64. MLAS implements:

- **SGEMM**: Single-precision general matrix multiply. It selects the block
  size for L1/L2/L3 cache tiling at runtime based on the matrix dimensions.
  For our MLP (small matrices, batch=1), MLAS uses a thin-panel algorithm.
  For large matrices it switches to a cache-blocked algorithm.
- **SCONV**: Winograd or direct convolution depending on kernel size.
- **AVX2/AVX512 dispatch**: MLAS detects CPU features at startup and
  dispatches to the widest SIMD register available.

### CUDAExecutionProvider

Dispatches to cuDNN for convolution and normalization, and cuBLAS for
matrix multiplications. It manages its own CUDA stream and a workspace
buffer for cuDNN algorithm scratch space.

ORT's CUDA EP also implements fused kernels for attention operations
(multi-head attention in one CUDA kernel) that are faster than the
equivalent sequence of cuBLAS calls.

### TensorRTExecutionProvider

Subgraphs compatible with TensorRT are handed to the TRT engine builder,
which:
1. Analyzes the subgraph
2. Selects optimal kernel implementations for the specific GPU architecture
   (e.g., Ampere SM86 vs. Hopper SM90)
3. Fuses layers beyond what ORT's graph optimizer does (e.g., multi-head
   attention + layer norm in a single kernel)
4. JIT-compiles CUDA kernels for those fused layers
5. Caches the resulting engine to disk

The first call is slow (compilation: several seconds to minutes). Subsequent
calls use the cached engine and are significantly faster than the plain CUDA
EP, especially for fixed batch sizes on production GPUs.

### Other EPs

| EP | Hardware | Backend |
|----|---------|---------|
| CoreMLExecutionProvider | Apple Silicon / iOS | Core ML framework |
| DirectMLExecutionProvider | Windows GPU (AMD/Intel/NVIDIA) | DirectX 12 |
| ROCmExecutionProvider | AMD GPU | MIOpen / rocBLAS |
| OpenVINOExecutionProvider | Intel CPU/GPU/VPU | Intel OpenVINO |
| NNAPIExecutionProvider | Android | Android NNAPI |
| AzureExecutionProvider | Azure ML | Remote inference |

---

## Numerical Fidelity: Why the Outputs Are Not Identical

Our comparison showed `max_abs_diff = 3.34e-06` between PyTorch and ORT on
the same model weights. This is not a bug. It comes from two sources:

**FP32 non-associativity**: floating-point addition is not associative.
`(a + b) + c ≠ a + (b + c)` in FP32 because intermediate results are
rounded after each operation. PyTorch's ATen SGEMM and ORT's MLAS SGEMM
may accumulate partial sums in different orders (different tile sizes,
different loop orderings), producing different rounding errors.

**FMA (Fused Multiply-Add)**: Modern CPUs and GPUs have FMA instructions
that compute `a × b + c` as a single rounded operation. If PyTorch uses
FMA in one path and ORT does not in another (or vice versa), the rounding
differs by one ULP (unit in the last place). At float32 precision, one ULP
at magnitude ~1.0 is ~1.2e-7.

For classification, these differences are far below the threshold that
changes the predicted class (argmax). For regression tasks with tight
numerical requirements, use `atol=1e-4` rather than `atol=1e-5` in your
acceptance test.

---

## What the ONNX Graph Looks Like for Our MLP

Running `inspect_graph()` on our exported MLP reveals exactly 8 nodes:

```
[00] Shape     (start=0, end=1)        input → batch_dim_tensor
[01] Concat    (axis=0)                batch_dim_tensor, [-1] → reshape_target
[02] Reshape   (allowzero=1)           input, reshape_target → view [B, 784]
[03] Gemm      (alpha=1, transB=1)     view, fc1.weight, fc1.bias → linear [B, 256]
[04] Relu                              linear → relu [B, 256]
[05] Gemm      (alpha=1, transB=1)     relu, fc2.weight, fc2.bias → linear_1 [B, 128]
[06] Relu                              linear_1 → relu_1 [B, 128]
[07] Gemm      (alpha=1, transB=1)     relu_1, fc3.weight, fc3.bias → logits [B, 10]
```

Nodes 0–2 implement `x.view(x.size(0), -1)` in a way that works for any
batch size:
- `Shape(start=0, end=1)` extracts dimension 0 of `input` (the batch size)
  as a 1-element tensor
- `Concat` builds a 2-element tensor `[batch_size, -1]`
- `Reshape` uses that tensor as the target shape

This pattern is how the ONNX exporter handles `view(-1)` with a dynamic
batch dimension. The `-1` in the reshape target is ONNX's way of saying
"infer this dimension from the total element count."

Nodes 3–8 are three `Gemm` + `Relu` pairs. Each `Gemm` has `transB=1`
because PyTorch stores `fc.weight` as `[out_features, in_features]` and
Gemm computes `C = alpha * A @ B^T + beta * C` — transposing B makes
the shapes compatible.

---

## Dynamic Axes vs Static Shapes

When `dynamic_axes=None` (the default), every dimension is frozen to the
dummy input's shape. Export with a batch-1 dummy means the ONNX model only
accepts batch=1 at runtime — ORT will raise a shape mismatch error for any
other batch size.

With `dynamic_axes={"input": {0: "batch_size"}, "logits": {0: "batch_size"}}`,
dimension 0 is stored as a symbolic `dim_param: "batch_size"` string. The
Shape → Concat → Reshape pattern is generated specifically to make the
flatten operation work for any batch size.

The tradeoff:
- **Dynamic batch** → flexible, slightly lower peak throughput (ORT cannot
  specialize memory planning for a fixed batch size)
- **Static batch** → rigid, slightly higher throughput, enables more
  aggressive TensorRT engine compilation

For server deployments where request batching is variable, use dynamic batch.
For edge deployments where batch=1 always, use static shapes for maximum
optimization.

---

## ONNX in the Compression Pipeline

ONNX is not a compression technique itself — it is the **export target** that
makes compressed models deployable:

```
Full FP32 model
    │
    │  1. Pruning (torch.nn.utils.prune)
    │     → sparse or compact architecture
    │
    ▼
Pruned model (still PyTorch)
    │
    │  2. Post-Training Quantization (torchao / torch.quantization)
    │     → INT8 weights, INT8 activations or INT8 weights, FP32 activations
    │
    ▼
Quantized model (still PyTorch)
    │
    │  3. ONNX Export (torch.onnx.export)
    │     → mlp_quantized.onnx (portable, no PyTorch required)
    │
    ▼
.onnx file
    │
    │  4. ONNX Runtime inference
    │     CPUExecutionProvider: serves on any CPU
    │     TensorRTExecutionProvider: maximum GPU throughput
    │
    ▼
Production inference
```

A quantized + pruned MLP exported to ONNX and served via ORT's CPU EP on
a modern x86 CPU is typically 4–8× faster than the equivalent PyTorch
eager-mode inference call, because:
1. No Python interpreter overhead
2. No autograd graph construction
3. MLAS SGEMM is more cache-efficient than ATen's default kernel for
   small batch inference
4. INT8 Gemm uses `vnni` instructions (AVX-512 VNNI / AMX) for 4× arithmetic
   throughput vs FP32 on Intel 3rd gen Xeon+

---

## Inspecting a Model with Netron

[Netron](https://netron.app) is a browser-based ONNX visualizer. Drag your
`.onnx` file onto the page to see an interactive graph. After shape inference,
every edge shows its tensor shape. Click any node to see its attributes and
the exact data type of each input/output.

This is the fastest way to verify that your export actually produced the
graph you expected.

---

## Common Export Issues

**Dynamic control flow not captured**
Symptom: the exported model always predicts the same class regardless of input.
Cause: a data-dependent branch was traced on one path only.
Fix: use `torch.onnx.dynamo_export` or restructure the branch to be tensor-
based (e.g., `torch.where` instead of Python `if`).

**Shape mismatch at runtime**
Symptom: `Invalid argument: Got invalid dimensions for input: ... Expected: [1, 784] Got: [32, 784]`
Cause: model exported without dynamic_axes, batch size differs at inference.
Fix: re-export with `dynamic_axes={"input": {0: "batch_size"}}`.

**Op not supported by target runtime**
Symptom: `No kernel registered for op: SomeOp`
Cause: using a custom or very new op that the target ORT version doesn't have.
Fix: lower the opset version, or find an equivalent op sequence.

**Numerical divergence > 1e-4**
Cause: usually quantization with the wrong zero point calibration, or a model
that uses double precision (float64) internally — ORT defaults to FP32.
Fix: check `model.double()` calls; ensure calibration dataset is representative.

---

## Summary

ONNX is a protobuf binary with a well-defined schema. The schema maps exactly
to what you see when you call `inspect_graph()`:
- `initializer[]` holds the weights as raw bytes
- `node[]` holds the ops as string-keyed records
- `input[]` / `output[]` / `value_info[]` hold tensor shapes

`torch.onnx.export` traces your model's forward pass, maps PyTorch ops to
ONNX symbolic functions, and serializes the result. The critical detail is
that tracing follows one execution path — dynamic control flow is invisible
to it.

ONNX Runtime loads that binary, runs graph optimization passes (constant
folding, node fusion, layout rewriting) at session creation time, partitions
the graph across execution providers, and builds a static execution plan
with pre-allocated memory. The result is a low-latency, framework-free
inference binary that runs on any hardware ORT supports.
