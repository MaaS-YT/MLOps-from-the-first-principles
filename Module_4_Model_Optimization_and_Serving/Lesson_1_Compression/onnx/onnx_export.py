# =============================================================================
# ONNX: Open Neural Network Exchange — Under the Hood
# =============================================================================
#
# WHAT ONNX IS AT ITS CORE
#
# ONNX is not a framework. It is a serialization format — a language-agnostic
# specification for describing a computation graph as a binary blob on disk.
# That blob is a Protocol Buffer (protobuf) binary following the schema
# defined in onnx/onnx.proto3.
#
# Every .onnx file is a serialized `ModelProto` message. Open any .onnx file
# in a hex editor and the first bytes are a protobuf field tag. The entire
# model — its architecture, its weights, its tensor shapes, its opset — lives
# in a single self-contained binary file.
#
# The protobuf hierarchy:
#
#   ModelProto                       ← the entire file
#   ├── ir_version: int              ← ONNX IR version (e.g., 8 for ONNX 1.13+)
#   ├── opset_import[]               ← which opset(s) the graph uses
#   │     └── OperatorSetIdProto
#   │           ├── domain: string   ← "" = standard ops, "com.microsoft" = custom
#   │           └── version: int     ← e.g., 18 means ONNX opset 18
#   └── graph: GraphProto            ← the actual computation
#         ├── node[]                 ← the operations (Conv, Relu, Gemm, ...)
#         │     └── NodeProto
#         │           ├── op_type    ← "Gemm", "Relu", "Reshape", ...
#         │           ├── input[]    ← names of tensors this op reads
#         │           ├── output[]   ← names of tensors this op writes
#         │           └── attribute[]← op-specific config (kernel_shape, etc.)
#         ├── input[]                ← model-level inputs (with shape/type)
#         ├── output[]               ← model-level outputs
#         ├── initializer[]          ← weight tensors (the actual float arrays)
#         │     └── TensorProto
#         │           ├── data_type  ← FLOAT=1, INT8=3, INT64=7, ...
#         │           ├── dims[]     ← shape, e.g. [256, 784]
#         │           └── raw_data   ← the bytes (row-major)
#         └── value_info[]           ← shapes of INTERMEDIATE tensors
#               (empty until shape inference runs)
#
# WHY THIS MATTERS
#
# Because the format is self-describing and framework-agnostic, a model trained
# in PyTorch on a Linux GPU can be loaded by ONNX Runtime on a Windows CPU,
# an ARM edge device, or a TensorRT cluster — with no Python, no PyTorch, no
# training dependencies. The consumer only needs the ONNX Runtime C library.
#
# =============================================================================

import os
import copy
import warnings
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import onnx
import onnx.helper as oh
import onnx.numpy_helper as onh
import onnxruntime as ort

warnings.filterwarnings("ignore")

# ---------------------------------------------------------------------------
# Reproducibility
# ---------------------------------------------------------------------------
SEED = 42
torch.manual_seed(SEED)
np.random.seed(SEED)
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device: {DEVICE}")

# Path for the .onnx file we will produce
ONNX_PATH = os.path.join(os.path.dirname(__file__), "mlp.onnx")


# =============================================================================
# SECTION 1: Model — the same MLP used throughout the compression section
# =============================================================================

class MLP(nn.Module):
    """Three-layer MLP for MNIST classification.

    Architecture: 784 → 256 → 128 → 10
    This is the baseline (no pruning, full FP32) that we export to ONNX.
    """
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(28 * 28, 256)
        self.fc2 = nn.Linear(256, 128)
        self.fc3 = nn.Linear(128, 10)

    def forward(self, x):
        # Flatten: (B, 1, 28, 28) → (B, 784)
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return self.fc3(x)


# =============================================================================
# SECTION 2: Train a baseline MLP on MNIST
# =============================================================================

def get_loaders(data_root: str, batch_size: int = 128):
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,)),
    ])
    train_ds = datasets.MNIST(root=data_root, train=True,  download=True, transform=transform)
    test_ds  = datasets.MNIST(root=data_root, train=False, download=True, transform=transform)
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True,  num_workers=2, pin_memory=True)
    test_loader  = DataLoader(test_ds,  batch_size=256,       shuffle=False, num_workers=2, pin_memory=True)
    return train_loader, test_loader


def train_one_epoch(model, loader, optimizer):
    model.train()
    total, correct, running_loss = 0, 0, 0.0
    for x, y in loader:
        x, y = x.to(DEVICE), y.to(DEVICE)
        optimizer.zero_grad()
        logits = model(x)
        loss = F.cross_entropy(logits, y)
        loss.backward()
        optimizer.step()
        running_loss += loss.item() * x.size(0)
        correct += (logits.argmax(1) == y).sum().item()
        total += x.size(0)
    return correct / total, running_loss / total


@torch.no_grad()
def evaluate(model, loader):
    model.eval()
    total, correct, running_loss = 0, 0, 0.0
    for x, y in loader:
        x, y = x.to(DEVICE), y.to(DEVICE)
        logits = model(x)
        loss = F.cross_entropy(logits, y)
        running_loss += loss.item() * x.size(0)
        correct += (logits.argmax(1) == y).sum().item()
        total += x.size(0)
    return correct / total, running_loss / total


def train_baseline(data_root: str, epochs: int = 3):
    train_loader, test_loader = get_loaders(data_root)
    model = MLP().to(DEVICE)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    for ep in range(epochs):
        tr_acc, tr_loss = train_one_epoch(model, train_loader, optimizer)
        te_acc, te_loss = evaluate(model, test_loader)
        print(f"  Epoch {ep+1}/{epochs} | train acc {tr_acc:.4f} loss {tr_loss:.4f} "
              f"| test acc {te_acc:.4f} loss {te_loss:.4f}")
    return model, test_loader


# =============================================================================
# SECTION 3: Export — what torch.onnx.export actually does
# =============================================================================
#
# HOW TRACING WORKS
#
# torch.onnx.export runs the model forward pass once with the dummy input
# in "tracing" mode. Under the hood it uses torch.jit.trace:
#
#   1. A trace graph is created by wrapping every tensor operation in a
#      `JitNode`. As Python executes model.forward(dummy), each op (matmul,
#      add, relu, etc.) records itself in the JIT IR graph instead of
#      executing immediately.
#
#   2. Once the forward pass is done, PyTorch walks the JIT IR graph and
#      maps each JIT op to its corresponding ONNX symbolic function.
#      These mapping functions live in torch/onnx/symbolic_opset*.py.
#      For example, F.relu → onnx::Relu, nn.Linear → onnx::Gemm (or
#      onnx::MatMul + onnx::Add depending on bias handling).
#
#   3. The ONNX symbolic functions emit NodeProto objects and TensorProto
#      initializers. The result is a fully populated GraphProto.
#
#   4. The GraphProto is wrapped in a ModelProto, serialized via protobuf,
#      and written to disk.
#
# WHAT TRACING CANNOT CAPTURE
#
#   - Data-dependent control flow: if tensor.item() > 0: ... The tracer
#     follows the branch taken during the single trace run. Alternative
#     branches are silently dropped.
#   - Python-level loops whose iteration count depends on a tensor value.
#   - In-place operations on non-leaf tensors (may silently corrupt the graph).
#
# If your model has these patterns, use torch.onnx.dynamo_export (TorchDynamo
# based, available in PyTorch ≥ 2.0) which captures the symbolic graph
# without executing it.
#
# OPSET VERSION
#
# The opset version controls which ONNX operator definitions are valid.
# Each ONNX release adds new ops or changes existing op semantics. Opset 18
# (ONNX 1.13) is the current standard tier. ORT supports opsets 7–22 as of
# ORT 1.17+. Always use the highest opset your target runtime supports to
# get the most complete operator coverage.
#
# DYNAMIC AXES
#
# By default, every tensor dimension is frozen to the shape of the dummy
# input. dynamic_axes tells the exporter to leave those dimensions symbolic
# (stored as a string name in TensorShapeProto.dim.dim_param instead of a
# fixed integer). This allows the ONNX Runtime to accept inputs of different
# batch sizes at inference time without re-exporting.
#
# =============================================================================

def export_to_onnx(model: nn.Module, onnx_path: str, opset: int = 18):
    model.eval()

    # The dummy input only sets the shape for tracing. Its actual values
    # don't matter for the graph structure (unless control flow depends on
    # them, which it doesn't here).
    dummy = torch.zeros(1, 1, 28, 28)

    print(f"\n[Export] Running torch.onnx.export → {onnx_path}")
    torch.onnx.export(
        model,
        dummy,
        onnx_path,
        input_names=["input"],
        output_names=["logits"],
        # batch_size=0 means "treat dimension 0 as symbolic (variable)"
        dynamic_axes={"input": {0: "batch_size"}, "logits": {0: "batch_size"}},
        opset_version=opset,
        verbose=False,
    )

    # -------------------------------------------------------------------------
    # Shape inference
    #
    # After export, value_info in the graph is empty — intermediate tensors
    # have no recorded shapes. onnx.shape_inference.infer_shapes() propagates
    # type and shape information through the graph by running each op's shape
    # inference function symbolically (no actual computation happens).
    #
    # After inference, every intermediate tensor gets a TypeProto entry in
    # value_info with its shape (possibly with symbolic batch dimensions).
    # This is required by some optimizers and visualization tools (Netron).
    # -------------------------------------------------------------------------
    onnx_model = onnx.load(onnx_path)
    onnx.checker.check_model(onnx_model)  # validates op types, input/output names, etc.
    inferred = onnx.shape_inference.infer_shapes(onnx_model)
    onnx.save(inferred, onnx_path)

    size_kb = os.path.getsize(onnx_path) / 1024
    print(f"[Export] Saved {onnx_path} ({size_kb:.1f} KB)")
    return inferred


# =============================================================================
# SECTION 4: Inspect the ONNX protobuf manually
# =============================================================================
#
# Every object we touch here is a Python object produced by protobuf
# deserialization. There is no magic — it is just nested structs. You can
# iterate, index, and print any field. This section shows you what the
# exporter actually produced.
#
# =============================================================================

def inspect_graph(onnx_model: onnx.ModelProto):
    print("\n" + "=" * 60)
    print("ONNX GRAPH INSPECTION")
    print("=" * 60)

    # -------------------------------------------------------------------------
    # ModelProto metadata
    # -------------------------------------------------------------------------
    print(f"\nIR version  : {onnx_model.ir_version}")
    print(f"Producer    : {onnx_model.producer_name} {onnx_model.producer_version}")
    for op_set in onnx_model.opset_import:
        domain = op_set.domain or "ai.onnx (standard)"
        print(f"Opset       : {domain} v{op_set.version}")

    graph = onnx_model.graph

    # -------------------------------------------------------------------------
    # Model-level inputs and outputs
    #
    # These are ValueInfoProto objects: they carry the tensor name, data type,
    # and shape (possibly with symbolic dims). The shape is stored as a
    # TensorShapeProto with a list of `dim` entries. Each dim is either a
    # fixed integer (dim_value) or a symbolic string (dim_param).
    # -------------------------------------------------------------------------
    print("\n--- Graph Inputs ---")
    for vi in graph.input:
        t = vi.type.tensor_type
        dtype = onnx.TensorProto.DataType.Name(t.elem_type)
        dims = [d.dim_param if d.HasField("dim_param") else d.dim_value
                for d in t.shape.dim]
        print(f"  {vi.name}: {dtype}{dims}")

    print("\n--- Graph Outputs ---")
    for vi in graph.output:
        t = vi.type.tensor_type
        dtype = onnx.TensorProto.DataType.Name(t.elem_type)
        dims = [d.dim_param if d.HasField("dim_param") else d.dim_value
                for d in t.shape.dim]
        print(f"  {vi.name}: {dtype}{dims}")

    # -------------------------------------------------------------------------
    # Initializers — the weight tensors
    #
    # TensorProto contains the raw weight bytes. `onh.to_array()` converts
    # them to a numpy array. The data is stored row-major (C order).
    # For large models these bytes dominate the .onnx file size.
    # -------------------------------------------------------------------------
    print(f"\n--- Initializers (weights) — {len(graph.initializer)} tensors ---")
    total_params = 0
    for init in graph.initializer:
        arr = onh.to_array(init)
        total_params += arr.size
        print(f"  {init.name:<40s}  shape={list(arr.shape)}  dtype={arr.dtype}")
    print(f"  Total parameters: {total_params:,}")

    # -------------------------------------------------------------------------
    # Nodes — the actual operations
    #
    # Each NodeProto encodes one op. `input` and `output` are lists of string
    # names that form the data-flow edges of the DAG. If an input name appears
    # in `initializer`, it is a constant weight. If it appears in another
    # node's output, it is a runtime-computed activation.
    # -------------------------------------------------------------------------
    print(f"\n--- Nodes ({len(graph.node)} ops) ---")
    for i, node in enumerate(graph.node):
        attrs = {a.name: _attr_value(a) for a in node.attribute}
        print(f"  [{i:02d}] {node.op_type:<12s}  "
              f"in={list(node.input)}  out={list(node.output)}"
              + (f"  attrs={attrs}" if attrs else ""))

    # -------------------------------------------------------------------------
    # Intermediate value shapes (populated by shape inference)
    # -------------------------------------------------------------------------
    if graph.value_info:
        print(f"\n--- Intermediate tensor shapes (after shape inference) ---")
        for vi in graph.value_info:
            t = vi.type.tensor_type
            dims = [d.dim_param if d.HasField("dim_param") else d.dim_value
                    for d in t.shape.dim]
            print(f"  {vi.name:<40s}  shape={dims}")

    print("=" * 60)


def _attr_value(attr):
    """Decode a protobuf AttributeProto to a readable Python value."""
    from onnx import AttributeProto as AP
    if attr.type == AP.FLOAT:    return attr.f
    if attr.type == AP.INT:      return attr.i
    if attr.type == AP.STRING:   return attr.s.decode()
    if attr.type == AP.FLOATS:   return list(attr.floats)
    if attr.type == AP.INTS:     return list(attr.ints)
    return f"<type={attr.type}>"


# =============================================================================
# SECTION 5: ONNX Runtime inference — sessions, optimization, providers
# =============================================================================
#
# WHAT HAPPENS WHEN YOU CREATE AN InferenceSession
#
#   Step 1 — Load & validate
#     ORT deserializes the ModelProto, validates op types against its
#     registered kernel catalogue, and checks input/output consistency.
#
#   Step 2 — Graph optimization (controlled by GraphOptimizationLevel)
#
#     DISABLED (0): graph is executed as-is. Useful for debugging.
#
#     BASIC (1): safe structural rewrites that cannot change numerics:
#       - Constant folding: subgraphs with only constant inputs are
#         evaluated at session creation and replaced with a single
#         Constant node. Example: a Reshape(x, [1, -1]) where -1 is
#         computed from a constant shape op gets folded.
#       - Redundant node elimination: Identity nodes, no-op Casts, etc.
#       - Common subexpression elimination: if two nodes compute the
#         same output, deduplicate them.
#
#     EXTENDED (2): hardware-specific fusions:
#       - Conv + BatchNormalization → ConvBatchNorm (merged weights)
#       - Conv + Relu → ConvRelu (kernel fusion)
#       - MatMul + Add → Gemm (single BLAS call)
#       - Gelu approximation fusion
#       - Layer/Instance norm fusion
#       These fusions replace multiple nodes with a single composite node
#       that maps to a single kernel call, eliminating memory round-trips.
#
#     ALL (3): layout optimization + EP-specific passes:
#       - NCHWc layout rewriting for CPU: reorders Conv weight memory layout
#         to match the CPU's preferred memory access pattern (AVX2/AVX512
#         channel-blocked loops). The physical weights change shape in memory
#         but compute the same values.
#       - NHWC rewriting for CUDA: similar for GPU cache alignment.
#
#   Step 3 — Partition across Execution Providers
#     ORT tries to assign each node to the highest-priority EP that has a
#     registered kernel for that op. Nodes without a GPU kernel fall back
#     to CPUExecutionProvider. Data movement nodes (MemcpyToHost,
#     MemcpyFromHost) are automatically inserted at EP boundaries.
#
#   Step 4 — Execution plan
#     A linear schedule of kernel calls is generated. Memory for intermediate
#     tensors is pre-allocated (or reused via memory planning) to minimize
#     runtime allocation overhead.
#
# EXECUTION PROVIDERS
#
#   CPUExecutionProvider
#     Uses MLAS (Microsoft Linear Algebra Subprograms) — a custom SIMD
#     library tuned for AVX2/AVX512. For Gemm it selects the best kernel
#     (SGEMM vs DGEMM, tiled vs blocked) based on matrix size at runtime.
#
#   CUDAExecutionProvider
#     Dispatches to cuDNN (conv, normalization) and cuBLAS (matmul).
#     Manages its own CUDA stream and workspace memory.
#
#   TensorRTExecutionProvider
#     Subgraphs compatible with TensorRT are handed to TRT's engine builder,
#     which JIT-compiles layer-fused kernels for the specific GPU architecture.
#     First inference is slow (compilation); subsequent calls use the cached
#     engine. Provides the best throughput on NVIDIA hardware.
#
#   Other EPs: CoreML (Apple), DirectML (Windows GPU), ROCm (AMD), OpenVINO
#     (Intel), NNAPI (Android).
#
# =============================================================================

def run_ort_inference(onnx_path: str, input_np: np.ndarray,
                      providers=None,
                      opt_level: ort.GraphOptimizationLevel = ort.GraphOptimizationLevel.ORT_ENABLE_EXTENDED):

    sess_opts = ort.SessionOptions()

    # Graph optimization level (see comment block above)
    sess_opts.graph_optimization_level = opt_level

    # intra_op_num_threads controls parallelism within a single op (e.g.,
    # how many threads MLAS uses for a matrix multiply).
    # 0 = ORT chooses based on hardware concurrency.
    sess_opts.intra_op_num_threads = 0

    if providers is None:
        providers = ["CPUExecutionProvider"]

    session = ort.InferenceSession(onnx_path, sess_opts, providers=providers)

    print(f"\n[ORT] Active providers: {session.get_providers()}")

    # Inspect what ORT sees as inputs/outputs after graph optimization.
    # These should match our exported model's input_names / output_names.
    for inp in session.get_inputs():
        print(f"  Input  '{inp.name}': shape={inp.shape}  type={inp.type}")
    for out in session.get_outputs():
        print(f"  Output '{out.name}': shape={out.shape}  type={out.type}")

    input_name = session.get_inputs()[0].name
    outputs = session.run(None, {input_name: input_np})
    return outputs, session


def compare_outputs(model: nn.Module, test_loader, onnx_path: str):
    print("\n" + "=" * 60)
    print("PYTORCH vs ONNX RUNTIME NUMERICAL COMPARISON")
    print("=" * 60)

    model.eval()
    x_batch, y_batch = next(iter(test_loader))
    x_batch = x_batch[:16]

    # PyTorch inference
    with torch.no_grad():
        pt_out = model(x_batch).cpu().numpy()

    # ONNX Runtime inference (CPU)
    x_np = x_batch.numpy()
    ort_outputs, _ = run_ort_inference(onnx_path, x_np)
    ort_out = ort_outputs[0]

    max_abs_diff = np.max(np.abs(pt_out - ort_out))
    mean_abs_diff = np.mean(np.abs(pt_out - ort_out))
    allclose = np.allclose(pt_out, ort_out, atol=1e-5, rtol=1e-4)

    print(f"\n  Max absolute difference  : {max_abs_diff:.2e}")
    print(f"  Mean absolute difference : {mean_abs_diff:.2e}")
    print(f"  np.allclose(atol=1e-5)   : {allclose}")

    # The tiny residuals come from:
    # 1. FP32 floating-point non-associativity — MLAS may accumulate sums in
    #    a different order than PyTorch's ATen kernels.
    # 2. AVX2/AVX512 fused-multiply-add (FMA) instructions: a*b+c is computed
    #    as a single rounded operation rather than two separate rounded ones.
    # For classification the predicted class (argmax) is always identical.

    pt_classes  = pt_out.argmax(axis=1)
    ort_classes = ort_out.argmax(axis=1)
    print(f"  Predicted classes match  : {np.all(pt_classes == ort_classes)}")
    print("=" * 60)


# =============================================================================
# SECTION 6: Show the effect of graph optimization levels
# =============================================================================
#
# For an MLP there are no Conv+BN fusions to see, but we can observe the
# effect on session creation time and inspect whether ORT simplified any nodes.
# On a full ResNet with BatchNorm this comparison is dramatic — the EXTENDED
# level folds BN parameters into the preceding Conv weights entirely.
#
# =============================================================================

def benchmark_opt_levels(onnx_path: str, input_np: np.ndarray):
    import time
    print("\n" + "=" * 60)
    print("ORT GRAPH OPTIMIZATION LEVEL BENCHMARK")
    print("=" * 60)

    levels = [
        ("DISABLED",  ort.GraphOptimizationLevel.ORT_DISABLE_ALL),
        ("BASIC",     ort.GraphOptimizationLevel.ORT_ENABLE_BASIC),
        ("EXTENDED",  ort.GraphOptimizationLevel.ORT_ENABLE_EXTENDED),
        ("ALL",       ort.GraphOptimizationLevel.ORT_ENABLE_ALL),
    ]

    for name, level in levels:
        # measure session creation (optimization cost) separately from inference
        t0 = time.perf_counter()
        opts = ort.SessionOptions()
        opts.graph_optimization_level = level
        opts.intra_op_num_threads = 1
        sess = ort.InferenceSession(onnx_path, opts, providers=["CPUExecutionProvider"])
        t_create = time.perf_counter() - t0

        # warm up
        input_name = sess.get_inputs()[0].name
        for _ in range(5):
            sess.run(None, {input_name: input_np})

        # measure 50 inference calls
        t1 = time.perf_counter()
        for _ in range(50):
            sess.run(None, {input_name: input_np})
        t_infer = (time.perf_counter() - t1) / 50 * 1000  # ms

        print(f"  {name:<12s}  session_create={t_create*1000:.1f}ms  "
              f"avg_infer={t_infer:.3f}ms")
    print("=" * 60)


# =============================================================================
# MAIN
# =============================================================================

if __name__ == "__main__":
    data_root = os.path.join(os.path.dirname(__file__), "data")

    # 1. Train
    print("\n[1] Training MLP on MNIST ...")
    model, test_loader = train_baseline(data_root, epochs=3)

    # Save FP32 weights alongside the .onnx file for reference
    pth_path = os.path.join(os.path.dirname(__file__), "mlp_fp32.pth")
    torch.save(model.state_dict(), pth_path)
    print(f"    Saved {pth_path}")

    # 2. Export
    print("\n[2] Exporting to ONNX ...")
    onnx_model = export_to_onnx(model.cpu(), ONNX_PATH, opset=18)

    # 3. Inspect the protobuf graph
    print("\n[3] Inspecting ONNX graph ...")
    inspect_graph(onnx_model)

    # 4. Run ORT inference and compare outputs
    print("\n[4] Comparing PyTorch vs ONNX Runtime outputs ...")
    # Move model to CPU for the comparison (ORT CPU provider)
    model_cpu = copy.deepcopy(model).cpu()
    compare_outputs(model_cpu, test_loader, ONNX_PATH)

    # 5. Benchmark optimization levels
    print("\n[5] Benchmarking ORT optimization levels ...")
    sample_np = torch.zeros(1, 1, 28, 28).numpy()
    benchmark_opt_levels(ONNX_PATH, sample_np)

    print("\nDone.")
