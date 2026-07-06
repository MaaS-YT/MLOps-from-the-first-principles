# Lesson 4.2b — LibTorch (C++ Inference)

| | |
|---|---|
| **Problem this solves** | Production systems — inference servers, robotics, game engines, embedded devices — are typically written in C++. Running Python just to call a PyTorch model is impractical: wrong language, too slow, too heavy. |
| **Mental model** | A TorchScript `.pt` file is a self-contained binary: computation graph + weights, no Python. The C++ LibTorch runtime loads it with `torch::jit::load()` and runs it natively — same weights, same results, zero Python. |
| **What the demo shows** | Train a binary XOR classifier in Python → export with `torch.jit.script` → load and run inference in C++ via LibTorch → correct predictions, no Python at runtime |
| **Where this fits** | Second step in the serving pipeline, after TorchScript compilation. This is how the exported model actually reaches a production C++ environment. |

---

## Files

| File | Purpose |
|------|---------|
| `LIBTORCH_GUIDE.md` | Full guide: the Python→C++ pipeline, IValue interface, CMake setup, CUDA toolkit notes |
| `train_and_save.py` | Train XOR model, export `cpp/model.pt` via `torch.jit.script` |
| `cpp/inference.cpp` | Load `model.pt`, run inference on all 4 XOR inputs, print results |
| `cpp/CMakeLists.txt` | CMake config linking against LibTorch from the installed torch package |
| `cpp/build.sh` | One-command build — handles the CUDA toolkit version workaround |

**Start with:** `LIBTORCH_GUIDE.md`

```bash
# Step 1 — train and export the model
uv run LibTorch/train_and_save.py

# Step 2 — build the C++ binary
bash LibTorch/cpp/build.sh

# Step 3 — run inference (no Python involved)
LD_LIBRARY_PATH=<venv>/torch/lib LibTorch/cpp/build/inference LibTorch/cpp/model.pt
```
