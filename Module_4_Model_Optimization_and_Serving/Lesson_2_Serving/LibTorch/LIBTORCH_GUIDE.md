# Lesson 4.2b — LibTorch (C++ Inference)

> **Lesson 4.2 / Part 2** — Load a TorchScript model in C++ and run inference without Python. The key insight: the `.pt` file produced by Python is a self-contained binary — the C++ runtime needs no knowledge of how the model was defined or trained.

| | |
|---|---|
| **Problem this solves** | Inference pipelines in production are almost never Python: they live inside C++ servers (gRPC, TensorRT), game engines (Unreal, Unity), robotics middleware (ROS), or embedded runtimes. Python is unavailable, too slow, or too memory-heavy in these environments. |
| **Mental model** | LibTorch is PyTorch's C++ API. It can load any `.pt` file, run any TorchScript graph, and manage tensors — without importing Python. The Python model is compiled once, and the C++ runtime runs it indefinitely. |
| **What the demo covers** | Train a binary XOR classifier → export as TorchScript → load in C++ → run inference → correct results for all four XOR inputs. |

---

## Part 1 — The Python → C++ Pipeline

### Overview

```
Python (training environment)          C++ (production environment)
─────────────────────────────          ────────────────────────────
  define model (nn.Module)
  train to convergence
  torch.jit.script(model)       ─────▶  model.pt  (self-contained binary)
  scripted.save("model.pt")             │
                                        ▼
                                  torch::jit::load("model.pt")
                                  module.eval()
                                  module.forward({input})
                                  → output tensor
```

The `.pt` file is the handoff point. Everything before it is Python. Everything after is C++.

---

### What is inside a .pt file

When `scripted.save("model.pt")` runs, it serialises three things into a single binary:

1. **The TorchScript IR graph** — a typed, platform-neutral representation of every operation in the model. No Python bytecode; no pickle of Python objects.
2. **All weight tensors** — serialised as raw data with dtype and shape metadata.
3. **Any constants** captured at script time — string literals, integer flags, etc.

The file format is based on a ZIP archive with a custom schema. It is stable across PyTorch versions and portable across operating systems and architectures.

---

### Why torch.jit.script (not trace) for production exports

`torch.jit.trace` also produces a `.pt` file, but it records only the single execution path taken during tracing. Any control flow that branches on a tensor value is silently baked in as the path taken during tracing (see TorchScript guide, Part 3 — The Silent Bug).

`torch.jit.script` compiles the full source. For production exports:
- Bugs from missed branches surface at export time, not at inference time
- The compiled graph is correct for all inputs, not just inputs similar to the trace example

---

## Part 2 — The C++ Side

### Key types and functions

#### `torch::jit::load(path)`

```cpp
torch::jit::script::Module module = torch::jit::load("model.pt");
```

Deserialises the `.pt` file: reconstructs the IR graph, allocates weight tensors on the device, and returns a `script::Module`. No Python, no pickle, no dynamic import.

---

#### `IValue` — the universal value container

LibTorch's `IValue` (Interpreted Value) is a tagged union that can hold any value a TorchScript program can produce: `Tensor`, `int`, `double`, `bool`, `string`, `List`, `Dict`, `Tuple`, or `None`.

`module.forward()` accepts `std::vector<IValue>` and returns `IValue`:

```cpp
std::vector<torch::jit::IValue> inputs;
inputs.push_back(input_tensor);           // wrap Tensor as IValue

torch::Tensor output = module.forward(inputs).toTensor();   // unwrap
```

The vector-of-IValue calling convention handles any model signature — single tensor, multiple tensors, mixed types — without needing to know the model's type signature in C++.

---

#### `torch::NoGradGuard`

```cpp
torch::NoGradGuard no_grad;
```

Equivalent to Python's `with torch.no_grad()`. It disables autograd for the duration of the enclosing scope. Without it, LibTorch allocates gradient metadata for every tensor operation — memory and compute wasted on inference. Always use it during inference.

---

#### `module.eval()`

```cpp
module.eval();
```

Switches `BatchNorm` and `Dropout` layers from training mode to inference mode — exactly as in Python. Omitting it does not cause an error, but it silently changes the statistical behaviour of any BatchNorm layers in the model.

---

### Tensor construction in C++

LibTorch tensors are created the same way as in Python — shape, dtype, device:

```cpp
// From a nested initializer list (like torch.tensor([[...], [...]]))
torch::Tensor input = torch::tensor(
    {{0.f, 0.f}, {0.f, 1.f}, {1.f, 0.f}, {1.f, 1.f}},
    torch::kFloat32
);

// Random tensor (like torch.randn)
torch::Tensor noise = torch::randn({4, 2});

// Zero tensor (like torch.zeros)
torch::Tensor z = torch::zeros({1, 2});
```

All standard tensor ops work identically: `.argmax()`, `.softmax()`, `.to(device)`, etc.

---

## Part 3 — CMake Setup

LibTorch ships its own CMake config (`TorchConfig.cmake`) that sets the correct include paths, library paths, and compile flags for your platform.

### Minimal CMakeLists.txt

```cmake
cmake_minimum_required(VERSION 3.18)
project(my_inference)

set(CMAKE_CXX_STANDARD 17)

find_package(Torch REQUIRED)   # reads TorchConfig.cmake

add_executable(inference inference.cpp)
target_link_libraries(inference "${TORCH_LIBRARIES}")
```

### Pointing CMake at LibTorch

```bash
cmake .. -DCMAKE_PREFIX_PATH=/path/to/libtorch/share/cmake
```

If you have PyTorch installed in a Python environment, you do not need a separate LibTorch download — the installed package bundles `TorchConfig.cmake` and all shared libraries:

```bash
# Get the CMake prefix path from the installed torch package
python -c "import torch; print(torch.utils.cmake_prefix_path)"
# → /your/venv/lib/python3.12/site-packages/torch/share/cmake
```

---

## Part 4 — CUDA Toolkit Notes

### How LibTorch finds CUDA

When CMake configures a LibTorch project, it searches for the system CUDA toolkit (`nvcc`) to verify version compatibility. The minimum required version is set by the PyTorch build (PyTorch 2.12 requires CUDA ≥ 12.1).

**This check is about the system CUDA toolkit used to build extensions, not about GPU availability at runtime.** If you are only running inference (no custom CUDA kernels), a CUDA ≥ 12.1 toolkit is only needed for CMake configuration — the GPU itself will work as long as the CUDA driver supports the runtime version bundled in the torch package.

### In this repo

The system `nvcc` is CUDA 12.0 but PyTorch 2.12 was compiled against CUDA 13.0. The mismatch causes CMake to fail the version check.

The workaround in `build.sh`:
1. Builds a fake CUDA root at `/tmp/fake_cuda13/` that contains the CUDA 13.0 headers from the `nvidia-cu13` Python package (installed alongside `torch`).
2. Points CMake at that root via `CUDA_TOOLKIT_ROOT_DIR`.
3. CMake finds headers saying `CUDA_VERSION 13000`, passes the version check, and proceeds.

No CUDA code is compiled — the workaround only satisfies the version check. All tensor operations use the pre-built LibTorch shared libraries.

---

## Part 5 — XOR Demo Results

### Training

```
Epoch  50 | val_acc = 1.000
Epoch 100 | val_acc = 1.000
Epoch 150 | val_acc = 1.000
Epoch 200 | val_acc = 1.000

Model saved to cpp/model.pt
Sample predictions (expect 1, 0, 0, 1): [1, 0, 0, 1]
```

### C++ inference output

```
Model loaded successfully.

Input:
 0  0
 0  1
 1  0
 1  1
[ CPUFloatType{4,2} ]

Logits:
  7.1  -6.9
 -9.0   8.2
 -5.1   8.4
  7.4  -6.4
[ CPUFloatType{4,2} ]

Predicted classes (expect 0, 1, 1, 0):
 0
 1
 1
 0
[ CPULongType{4} ]
```

All four XOR inputs are correctly classified by the C++ binary. The logits show high confidence: large positive values on the correct class, large negatives on the wrong class.

---

## Part 6 — TorchScript vs LibTorch: Where Each Lives

| | TorchScript | LibTorch |
|---|---|---|
| Language | Python (export side) | C++ (runtime side) |
| Role | Compile the model | Run the compiled model |
| Key API | `torch.jit.script`, `scripted.save()` | `torch::jit::load`, `module.forward()` |
| Output | `.pt` file | Inference result |
| Python required | Yes — this is still Python | No — pure C++ |
| When to use | At export time, once | At inference time, in production |

They are two halves of the same workflow. TorchScript without LibTorch gives you a portable file you can't use in C++. LibTorch without TorchScript gives you a C++ runtime with nothing to load.

---

## Quick Reference

### Python export

```python
model.eval()
scripted = torch.jit.script(model)
scripted.save("model.pt")
```

### C++ inference skeleton

```cpp
#include <torch/script.h>

torch::jit::script::Module model = torch::jit::load("model.pt");
model.eval();

torch::NoGradGuard no_grad;
std::vector<torch::jit::IValue> inputs = {your_tensor};
torch::Tensor output = model.forward(inputs).toTensor();
```

### Build

```bash
cmake .. -DCMAKE_PREFIX_PATH=$(python -c "import torch; print(torch.utils.cmake_prefix_path)")
cmake --build . --config Release
```

---

## Official Documentation

- LibTorch C++ API: https://pytorch.org/cppdocs/
- Loading a TorchScript model in C++: https://pytorch.org/tutorials/advanced/cpp_export.html
- torch::jit::load: https://pytorch.org/cppdocs/api/function_namespacetorch_1_1jit_1a66c578f5d7b46e01cfc8aad7a7f2e9a2.html
- IValue API: https://pytorch.org/cppdocs/api/structc10_1_1IValue.html
