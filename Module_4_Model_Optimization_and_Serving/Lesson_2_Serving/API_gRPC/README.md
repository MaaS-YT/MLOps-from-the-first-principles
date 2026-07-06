# Lesson 4.2c — gRPC ML Serving

| | |
|---|---|
| **Problem this solves** | REST/HTTP is text-based and stateless — every call re-parses JSON and re-establishes context. For high-throughput ML inference (thousands of requests per second, streaming predictions, microservice-to-microservice calls) this overhead is unacceptable. |
| **Mental model** | gRPC uses binary Protobuf messages over a persistent HTTP/2 connection. The `.proto` file is the contract: both server and client generate their code from it, so the interface is always in sync. Calling a remote model feels like calling a local function. |
| **What the demo shows** | Define a Predict RPC in a `.proto` file → generate Python server/client stubs → train a linear regression model → serve it over gRPC → call it from a client and receive structured binary responses. Optionally deploy the server in Docker. |
| **Where this fits** | Alternative serving layer to REST APIs. Use gRPC when latency, throughput, or inter-service communication is the constraint — REST when human-readable APIs or browser clients are needed. |

---

## Files

| File | Purpose |
|------|---------|
| `GRPC_GUIDE.md` | Full guide: why gRPC, Protobuf syntax, code generation, server/client architecture, Docker, gRPC vs REST |
| `prediction.proto` | Service contract — defines the `Predict` RPC, request and response message types |
| `train.py` | Train a linear regression model, save to `model.pkl` |
| `server.py` | Load `model.pkl`, implement the `MLModelServicer`, start the gRPC server on port 5000 |
| `client.py` | Open a channel, call `Predict` for four test inputs, print structured responses |
| `Dockerfile` | Containerise the server — copies model, generates stubs inside the image, exposes port 5000 |
| `pyproject.toml` | Isolated environment for this demo — **must stay separate from the root env** (see note below) |

> **Why a separate environment?**
> `grpcio-tools 1.75` requires `protobuf >= 6.31`, but the root DDODS env pins
> `protobuf >= 4.24, < 5` (required by feast). These are incompatible — merging
> would break the root env's dependency resolution. Always run this demo from
> inside the `API_gRPC/` directory using its own `uv sync` / `.venv`.

**Start with:** `GRPC_GUIDE.md`

---

## Quick Start (local, no Docker)

```bash
# 1. Move into the demo directory and install dependencies
cd Module_4_Model_Optimization_and_Serving/Lesson_2_Serving/API_gRPC
uv sync

# 2. Train and save the model
uv run train.py

# 3. Generate gRPC Python stubs from the .proto file
#    This creates prediction_pb2.py and prediction_pb2_grpc.py
uv run python -m grpc_tools.protoc -I. --python_out=. --grpc_python_out=. prediction.proto

# 4. Start the server — keep this terminal open
uv run server.py
```

Open a **second terminal**, then:

```bash
cd Module_4_Model_Optimization_and_Serving/Lesson_2_Serving/API_gRPC
uv run client.py
```

**Expected output (server terminal):**
```
Model loaded successfully
Server started — listening on port 5000
  request: [1.0] → 2.0000
  request: [3.5] → 7.0000
  request: [5.0] → 10.0000
  request: [10.0] → 20.0000
```

**Expected output (client terminal):**
```
Connecting to gRPC server at localhost:5000...

Input      : 1.0
Prediction : 2.00
Model ver  : v1.0
------------------------------
Input      : 3.5
Prediction : 7.00
Model ver  : v1.0
------------------------------
Input      : 5.0
Prediction : 10.00
Model ver  : v1.0
------------------------------
Input      : 10.0
Prediction : 20.00
Model ver  : v1.0
------------------------------
```

---

## Quick Start (Docker)

```bash
# 1. Move into the demo directory and train the model
#    (Docker copies model.pkl into the image at build time)
cd Module_4_Model_Optimization_and_Serving/Lesson_2_Serving/API_gRPC
uv run train.py

# 2. Build the image — stubs are generated inside the container
docker build -t grpc-server .

# 3. Run the container, mapping host port 5000 → container port 5000
docker run -p 5000:5000 grpc-server
```

Open a **second terminal**, then:

```bash
cd Module_4_Model_Optimization_and_Serving/Lesson_2_Serving/API_gRPC
uv run client.py
```

Expected output is the same as the local run above.
