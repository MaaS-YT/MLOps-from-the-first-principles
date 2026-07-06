/*
 * LibTorch C++ inference demo
 *
 * The core idea:
 *   A TorchScript .pt file produced by Python's scripted.save() is a
 *   self-contained binary: it carries the IR graph and trained weights.
 *   torch::jit::load() deserialises it entirely in C++ — no Python interpreter,
 *   no pickle, no dynamic imports.  This is the standard path for deploying
 *   PyTorch models to production servers, games, robotics, or mobile apps.
 *
 * This file demonstrates:
 *   1. Load a TorchScript module with torch::jit::load()
 *   2. Construct a batch input tensor in C++
 *   3. Run forward() via the IValue interface
 *   4. Extract the predicted class with argmax
 *
 * Build:
 *   bash build.sh           (handles the CUDA version workaround automatically)
 *
 * Run:
 *   LD_LIBRARY_PATH=<venv>/torch/lib ./build/inference ../model.pt
 */

#include <torch/script.h>   // one-stop header for TorchScript / LibTorch
#include <iostream>
#include <vector>

int main(int argc, const char* argv[]) {
    if (argc != 2) {
        std::cerr << "Usage: inference <path/to/model.pt>\n";
        return 1;
    }

    // -----------------------------------------------------------------------
    // 1. Load the TorchScript module
    //
    // torch::jit::load() is the C++ counterpart of Python's torch.jit.load().
    // It reads the .pt file, reconstructs the IR graph, and allocates the
    // weight tensors — all without touching Python or the pickle protocol.
    //
    // model.eval() switches BatchNorm and Dropout layers to inference mode,
    // exactly as in Python.  Omitting it leaves training behaviour active,
    // which silently changes the output statistics.
    // -----------------------------------------------------------------------
    torch::jit::script::Module model;
    try {
        model = torch::jit::load(argv[1]);
        model.eval();   // disable dropout / batchnorm training behaviour
    } catch (const c10::Error& e) {
        std::cerr << "Error loading model: " << e.what() << "\n";
        return 1;
    }
    std::cout << "Model loaded successfully.\n\n";

    // -----------------------------------------------------------------------
    // 2. Build the four XOR input combinations as a single batch tensor
    //    Shape: [4, 2]  (4 samples, 2 features each)
    // -----------------------------------------------------------------------
    torch::Tensor input = torch::tensor(
        {{0.f, 0.f},
         {0.f, 1.f},
         {1.f, 0.f},
         {1.f, 1.f}},
        torch::kFloat32
    );

    // -----------------------------------------------------------------------
    // 3. Run inference
    //
    // IValue (Interpreted Value) is LibTorch's tagged-union type — it can hold
    // a Tensor, int, bool, string, list, dict, or None.  forward() accepts a
    // std::vector<IValue> so a single calling convention handles models with
    // any mix of input types.
    //
    // torch::NoGradGuard disables gradient tracking for the duration of this
    // scope, equivalent to Python's `with torch.no_grad()`.  Without it,
    // LibTorch allocates autograd metadata for every tensor operation, wasting
    // memory and compute that is only needed during training.
    // -----------------------------------------------------------------------
    std::vector<torch::jit::IValue> inputs;
    inputs.push_back(input);

    torch::NoGradGuard no_grad;   // equivalent to torch.no_grad() in Python
    torch::Tensor logits = model.forward(inputs).toTensor();  // shape [4, 2]
    torch::Tensor preds  = logits.argmax(1);                  // shape [4]

    // -----------------------------------------------------------------------
    // 4. Print results
    // -----------------------------------------------------------------------
    std::cout << "Input:\n"   << input  << "\n\n";
    std::cout << "Logits:\n"  << logits << "\n\n";
    std::cout << "Predicted classes (expect 0, 1, 1, 0):\n" << preds << "\n";

    return 0;
}
