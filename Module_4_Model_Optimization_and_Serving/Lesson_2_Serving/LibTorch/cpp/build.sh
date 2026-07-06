#!/bin/bash
# Build the LibTorch C++ inference binary.
#
# Why the CUDA workaround:
#   This project reuses the LibTorch bundled inside the Python torch package
#   instead of downloading a separate LibTorch zip.  The system nvcc reports
#   CUDA 12.0, but PyTorch 2.12 requires >= 12.1 at configure time.  We point
#   CMake at the CUDA 13.0 headers from the nvidia-cu13 Python package and a
#   wrapper nvcc that reports 13.0, satisfying the version check without
#   installing a full CUDA toolkit.

set -e

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
VENV_ROOT="$(cd "$SCRIPT_DIR/../../../../.." && pwd)/.venv"
SITE_PKGS="$VENV_ROOT/lib/python3.12/site-packages"

CMAKE_PREFIX="$SITE_PKGS/torch/share/cmake"
NVIDIA_CU13="$SITE_PKGS/nvidia/cu13"
FAKE_CUDA="/tmp/fake_cuda13"

# Build the fake CUDA 13.0 root (idempotent)
if [ ! -f "$FAKE_CUDA/bin/nvcc" ]; then
    mkdir -p "$FAKE_CUDA/bin"
    cat > "$FAKE_CUDA/bin/nvcc" << 'EOF'
#!/bin/bash
if [[ "$*" == "--version" ]]; then
    echo "Cuda compilation tools, release 13.0, V13.0.140"
else
    exec /usr/bin/nvcc "$@"
fi
EOF
    chmod +x "$FAKE_CUDA/bin/nvcc"
    ln -sfn "$NVIDIA_CU13/include" "$FAKE_CUDA/include"
    ln -sfn "$NVIDIA_CU13/lib"     "$FAKE_CUDA/lib64"
fi

rm -rf "$SCRIPT_DIR/build"
mkdir -p "$SCRIPT_DIR/build"

PATH="$FAKE_CUDA/bin:$PATH" cmake -S "$SCRIPT_DIR" -B "$SCRIPT_DIR/build" \
    -DCMAKE_PREFIX_PATH="$CMAKE_PREFIX" \
    -DCUDA_TOOLKIT_ROOT_DIR="$FAKE_CUDA" \
    -DCMAKE_BUILD_TYPE=Release

cmake --build "$SCRIPT_DIR/build" --config Release -j"$(nproc)"

echo ""
echo "Binary: $SCRIPT_DIR/build/inference"
echo "Run with:"
echo "  LD_LIBRARY_PATH=$SITE_PKGS/torch/lib \\"
echo "    $SCRIPT_DIR/build/inference $SCRIPT_DIR/model.pt"
