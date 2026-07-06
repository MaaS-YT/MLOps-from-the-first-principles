# Lesson 4.2 — Model Serving

| | |
|---|---|
| **Problem this solves** | A trained PyTorch model is coupled to Python, to the training framework, and to the machine it was trained on. Serving that model in production — in a C++ server, a mobile app, or a latency-critical pipeline — requires decoupling the computation from the Python runtime. |
| **Mental model** | Compilation is the bridge: export the model's computation graph once (TorchScript), then load and run it anywhere the LibTorch runtime exists — C++, mobile, or embedded — without Python. |
| **What the lesson demonstrates** | Three deployment paths: TorchScript compiles a model to portable IR → LibTorch runs it in C++ without Python → gRPC serves it over a binary RPC interface to any language. |
| **Where this fits** | This is the **Serving Layer** in the system map. It follows compression (Lesson 4.1) and answers the question: once the model is small enough, how do you actually deploy it? |

---

## Structure

```
Lesson_2_Serving/
  TorchScript/
    TORCHSCRIPT_GUIDE.md   ← trace vs script, silent bugs, benchmark
    torchJIT.py            ← trace demo → silent bug → script fix
  LibTorch/
    LIBTORCH_GUIDE.md      ← Python export → C++ inference pipeline
    train_and_save.py      ← train XOR model, export model.pt
    cpp/
      inference.cpp        ← load model.pt and run inference in C++
      CMakeLists.txt       ← LibTorch CMake config
      build.sh             ← one-command build (handles CUDA workaround)
  API_gRPC/
    GRPC_GUIDE.md          ← why gRPC, Protobuf, code generation, server/client, Docker
    prediction.proto       ← service contract (source of truth)
    train.py               ← train linear regression, save model.pkl
    server.py              ← gRPC server implementing the Predict RPC
    client.py              ← gRPC client calling the server
    Dockerfile             ← containerised server
```

**Read order:** `TorchScript/TORCHSCRIPT_GUIDE.md` → `LibTorch/LIBTORCH_GUIDE.md` → `API_gRPC/GRPC_GUIDE.md`

---

## The Serving Pipeline

```
PyTorch model  (eager, Python-bound)
      │
      ▼  torch.jit.script()   — compile Python source → TorchScript IR
TorchScript module
      │
      ▼  scripted.save()      — serialise graph + weights → model.pt
model.pt  (self-contained binary, no Python needed to run)
      │
      ├──▶  torch::jit::load()   — C++ LibTorch runtime
      │     C++ inference  →  production server / mobile / embedded
      │
      └──▶  gRPC server          — serve over binary RPC interface
            prediction.proto     — typed contract for any language client
            Docker container  →  deploy anywhere
```
