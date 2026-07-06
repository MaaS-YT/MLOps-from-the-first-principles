# Lesson 4.2c — gRPC ML Serving

> **Lesson 4.2 / Part 3** — Serve a machine learning model over gRPC using Protobuf contracts and HTTP/2. The key insight: define the interface once in a `.proto` file, generate client and server code from it, and get type-safe binary communication for free.

| | |
|---|---|
| **Problem this solves** | REST APIs parse JSON text on every call, re-establish HTTP/1.1 connections, and have no enforced schema. For ML inference pipelines — where throughput, latency, and inter-service contracts matter — this is inefficient and fragile. |
| **Mental model** | gRPC is an RPC framework: calling a remote model method looks and feels like a local function call. The `.proto` contract defines the messages and methods; both sides generate code from it so the interface is always in sync. Binary serialisation and HTTP/2 multiplexing make it fast. |
| **What the demo covers** | Write a `.proto` contract → generate stubs → train a model → implement the server → call it from a client → containerise with Docker. |

---

## Part 1 — Why gRPC for ML Serving

### The REST bottleneck

REST over HTTP/1.1 is the default choice for web APIs, but it carries overhead that accumulates fast in ML inference pipelines:

```
REST / HTTP 1.1 call:
  client → serialise dict to JSON string
         → open TCP connection (or reuse from pool)
         → send HTTP/1.1 text headers + JSON body
         → server parses JSON text → dict → numpy array
         → run model
         → convert result → dict → JSON string
         → send HTTP response
         → client parses JSON string → dict
```

Every step involves text parsing, schema-less dicts, and repeated connection overhead.

### gRPC over HTTP/2

```
gRPC call:
  client → serialise PredictRequest to Protobuf binary (~5 bytes for one float)
         → send over existing HTTP/2 stream (no new TCP handshake)
         → server deserialises binary → typed PredictRequest object
         → run model
         → serialise PredictResponse to binary
         → send over same stream
         → client deserialises → typed PredictResponse object
```

Key differences:

| | REST / JSON | gRPC / Protobuf |
|---|---|---|
| Serialisation | Text (JSON) | Binary (Protobuf) |
| Schema | Optional (OpenAPI) | Enforced (`.proto`) |
| Transport | HTTP/1.1 (one request per connection) | HTTP/2 (many streams per connection) |
| Code generation | Manual | Automatic from `.proto` |
| Browser support | Native | Needs gRPC-web proxy |
| Human readable | Yes | No (use `grpcurl` or `grpc_cli` to inspect) |
| Typical latency | Higher (text parse + new connections) | Lower (binary + multiplexed) |
| Streaming | Limited (websockets hack) | First-class (server, client, bidirectional) |

---

### Where gRPC fits in the serving stack

```
Training (Python, eager)
      │
      ▼  Compression  (Lesson 4.1)
Optimised model
      │
      ├──▶  TorchScript + LibTorch  (Lesson 4.2a/b)   — C++ runtime, no Python
      │
      └──▶  gRPC serving  (Lesson 4.2c)   ← you are here
               │
               ▼
         Microservice interface — other services call Predict() like a local function
```

Use gRPC when:
- Calling the model from another service (not from a browser)
- Latency or throughput is a hard constraint
- You want a typed, versioned contract enforced at the API level
- You need streaming (returning predictions for a stream of inputs)

Use REST when:
- The consumer is a browser or a human
- Simplicity and debuggability matter more than raw performance
- You need to expose the API publicly with standard tooling

---

## Part 2 — gRPC vs FastAPI: Why Both Exist in This Course

This course teaches FastAPI in Lesson 1.1 and gRPC here. They both serve ML models
over a network — so why learn two tools?

The answer is that they solve different problems in different parts of the system.

### What FastAPI is built for

FastAPI wraps a Python function behind an HTTP endpoint. It is designed for:

- **Human-facing APIs** — browsers, mobile apps, curl, Postman all speak HTTP/JSON natively
- **Public APIs** — external developers expect REST; it is the universal contract
- **Rapid development** — one decorator, automatic Swagger docs, no code generation
- **Simple request patterns** — one client sends one request, gets one response

```python
@app.post("/predict")
def predict(input: InputSchema) -> OutputSchema:
    return model(input)
```

This is exactly right for Lesson 1.1: a data scientist wraps a trained model and exposes
it so anything can call it. The simplicity is the point.

---

### What gRPC is built for

gRPC is a contract-first RPC framework. It is designed for:

- **Service-to-service communication** — one backend calls another; no human is involved
- **High-throughput pipelines** — thousands of inference requests per second
- **Typed contracts** — the `.proto` file is the schema; both sides generate code from it
- **Streaming** — server-side, client-side, or bidirectional — REST has no clean equivalent
- **Multi-language systems** — Python model server called by a Go API gateway or C++ client

```protobuf
service MLModel {
  rpc Predict(PredictRequest) returns (PredictResponse);
  rpc PredictStream(stream PredictRequest) returns (stream PredictResponse);
}
```

---

### The real difference: who is calling the model

```
Browser / mobile app / external developer
        │
        ▼
  FastAPI  (Lesson 1.1)
  REST/JSON over HTTP/1.1
  Human-readable, standard, easy to debug
        │
        ▼
  Internal ML model server
        │
        ▼
  gRPC  (this lesson)
  Binary Protobuf over HTTP/2
  Fast, typed, streaming-capable
        │
        ▼
  Another internal service (recommendation engine, fraud model, etc.)
```

In production, a system often has both running simultaneously:
- FastAPI handles the external user-facing endpoint (the "front door")
- gRPC handles the internal calls between model servers, feature pipelines, and downstream services (the "plumbing")

---

### Side-by-side comparison

| | FastAPI | gRPC |
|---|---|---|
| Protocol | HTTP/1.1 | HTTP/2 |
| Payload format | JSON (text) | Protobuf (binary) |
| Schema | Optional (Pydantic) | Mandatory (`.proto`) |
| Code generation | None needed | Required (`protoc`) |
| Browser support | Native | Needs gRPC-web proxy |
| Streaming | Limited | First-class |
| Typical latency | Higher (text parse) | Lower (binary) |
| Debugging | Easy (curl, browser) | Needs `grpcurl` |
| Setup complexity | Low | Medium |
| **Best for** | Public APIs, browser clients | Microservices, high-throughput, internal RPC |

---

### When to choose which

**Choose FastAPI when:**
- Your client is a browser, mobile app, or external developer
- You want auto-generated Swagger docs
- You need to ship quickly and the load is moderate
- Human readability and debuggability matter more than raw performance

**Choose gRPC when:**
- Service A is calling Service B — both are backend services you control
- You need thousands of requests per second or streaming predictions
- You want a strictly typed, versioned contract that both sides generate code from
- You are building a multi-language system (Python model, Go gateway, C++ client)

**Use both when:**
- You have a public REST API (FastAPI) that internally calls model servers over gRPC
- This is the standard production ML architecture: REST at the edge, gRPC in the interior

---

## Part 3 — Protocol Buffers


### The .proto file

`prediction.proto` is the single source of truth for this service. It defines:
1. The **message types** — the structured data that flows over the wire
2. The **service** — the RPC methods the server exposes

```protobuf
syntax = "proto3";

// The service block defines the RPC interface.
// Each `rpc` line becomes a method on the server Servicer and client Stub.
service MLModel {
  rpc Predict(PredictRequest) returns (PredictResponse);
}

// Messages are like typed structs.
// Field numbers (= 1, = 2) are used for binary encoding — they must never change
// once a field is in production (changing them breaks backward compatibility).
message PredictRequest {
  repeated float features = 1;   // `repeated` = variable-length list, like List[float]
}

message PredictResponse {
  float  prediction    = 1;
  string model_version = 2;
}
```

### Scalar types

| Proto type | Python type | Notes |
|---|---|---|
| `float` | `float` | 32-bit IEEE 754 |
| `double` | `float` | 64-bit IEEE 754 |
| `int32` | `int` | |
| `int64` | `int` | |
| `bool` | `bool` | |
| `string` | `str` | UTF-8 |
| `bytes` | `bytes` | raw binary — useful for passing serialised tensors |

### Field numbers

Field numbers (e.g., `= 1`, `= 2`) are the binary identity of each field. The field name is discarded at serialisation time — only the number is encoded. This means:

- You **can** rename a field without breaking compatibility
- You **cannot** change a field number — existing clients will misread the message
- You **cannot** reuse a field number once it has been published, even after deletion

---

## Part 4 — Code Generation

The `.proto` file is not Python — it must be compiled into Python stubs before the server or client can use it.

```bash
python -m grpc_tools.protoc \
    -I.                          \   # search path for .proto imports
    --python_out=.               \   # where to write the Protobuf message classes
    --grpc_python_out=.          \   # where to write the gRPC service stubs
    prediction.proto
```

This generates two files:

### `prediction_pb2.py`

Contains the Python classes for every message defined in the `.proto` file:

```python
# Auto-generated — do not edit manually
class PredictRequest(Message):
    features: List[float]

class PredictResponse(Message):
    prediction: float
    model_version: str
```

### `prediction_pb2_grpc.py`

Contains:
- `MLModelStub` — the **client-side** proxy class. Calling `stub.Predict(request)` serialises the request, sends it, and returns the deserialised response.
- `MLModelServicer` — the **server-side** base class with abstract methods for each RPC. You subclass this and implement the methods.
- `add_MLModelServicer_to_server()` — registers a servicer implementation with a `grpc.Server` instance.

**Never edit these files manually.** Regenerate them if the `.proto` file changes.

---

## Part 5 — Server Architecture

### The Servicer

```python
class MLModelService(prediction_pb2_grpc.MLModelServicer):

    def __init__(self, model_path="model.pkl"):
        self.model = joblib.load(model_path)   # load once at startup

    def Predict(self, request, context):
        features   = np.array(request.features).reshape(1, -1)
        prediction = float(self.model.predict(features)[0])
        return prediction_pb2.PredictResponse(
            prediction=prediction,
            model_version="v1.0",
        )
```

`request` is a `PredictRequest` instance — fields are typed Python attributes, not dict keys.

`context` is a `ServicerContext` — use it to set error status if the call fails:

```python
context.set_code(grpc.StatusCode.INTERNAL)
context.set_details("Model failed: division by zero")
return prediction_pb2.PredictResponse()   # empty response on error
```

### The Server

```python
server = grpc.server(futures.ThreadPoolExecutor(max_workers=10))
prediction_pb2_grpc.add_MLModelServicer_to_server(MLModelService(), server)
server.add_insecure_port("[::]:5000")
server.start()
server.wait_for_termination()
```

- `ThreadPoolExecutor(max_workers=10)` — each incoming RPC is dispatched to a thread. Tune `max_workers` to: `expected_concurrency * (inference_latency_s / acceptable_queue_time_s)`.
- `[::]:5000` — binds all network interfaces on port 5000 (both IPv4 and IPv6). Use `0.0.0.0:5000` for IPv4-only.
- `add_insecure_port` — no TLS. For production, use `add_secure_port` with `grpc.ssl_server_credentials()`.
- `wait_for_termination()` — blocks the main thread. Without it the process exits immediately after `start()`.

---

## Part 6 — Client Architecture

### Channel and stub

```python
with grpc.insecure_channel("localhost:5000") as channel:
    stub = prediction_pb2_grpc.MLModelStub(channel)
    response = stub.Predict(PredictRequest(features=[3.5]))
```

- **Channel** — an HTTP/2 connection to the server. Creating a channel is expensive; reuse it across all calls. The `with` block ensures it is closed on exit.
- **Stub** — a thin proxy that wraps the channel. Calling `stub.Predict()` is a blocking call that serialises the request, sends it, and returns the deserialised response.
- **Deadline** — add `timeout=5.0` to any stub call to enforce a maximum wait time. Without it, a hanging server will block the client indefinitely.

```python
response = stub.Predict(request, timeout=5.0)
```

### Error handling

```python
try:
    response = stub.Predict(request)
except grpc.RpcError as e:
    print(f"Status: {e.code()}")     # e.g. StatusCode.INTERNAL
    print(f"Detail: {e.details()}")  # the string from context.set_details()
```

gRPC status codes map to familiar HTTP concepts: `NOT_FOUND` (404), `INTERNAL` (500), `UNAVAILABLE` (503 — server not running), `DEADLINE_EXCEEDED` (timeout).

---

## Part 7 — Docker Deployment

### What the Dockerfile does

```dockerfile
FROM python:3.12-slim
WORKDIR /app

# Copy source files including the pre-trained model.pkl
COPY server.py model.pkl prediction.proto requirements.txt /app/

# Install gRPC dependencies
RUN pip install -r requirements.txt

# Generate stubs inside the container from the .proto file
RUN python -m grpc_tools.protoc -I. --python_out=. --grpc_python_out=. prediction.proto

EXPOSE 5000
CMD ["python", "-u", "server.py"]
```

The stubs are generated **inside** the container during build — they are not copied from the host. This guarantees the stubs always match the `.proto` file that ships with the image.

### Why `model.pkl` must exist before `docker build`

The `COPY` instruction copies `model.pkl` from the build context (the host). If it does not exist, the build fails. Always run `uv run train.py` before building the image.

### Port mapping

```bash
docker run -p 5000:5000 grpc-server
#              ^^^^  ^^^^
#              host  container
```

The client always connects to `localhost:5000` (host port). Docker forwards it to the container's port 5000 where the server is listening.

---

## Part 8 — gRPC vs REST: When to Use Which

| Criterion | REST / JSON | gRPC / Protobuf |
|---|---|---|
| Payload size | Large (verbose JSON) | Small (compact binary) |
| Latency | Higher | Lower |
| Throughput | Lower | Higher |
| Schema enforcement | Optional | Mandatory (`.proto`) |
| Browser clients | Native | Needs gRPC-web |
| Streaming | Workaround needed | Native (4 modes) |
| Debugging / inspection | Easy (curl, browser) | Needs `grpcurl` or `grpc_cli` |
| Versioning | Manual (URL version, header) | Built in (field numbers) |
| **Choose when** | Public API, browser clients, simplicity | Microservices, high throughput, typed contracts |

---

## Part 9 — The Three-File Pipeline

```
train.py          →  model.pkl
                                 \
prediction.proto  →  protoc  →  prediction_pb2.py
                                 prediction_pb2_grpc.py
                                        │
                          ┌─────────────┴─────────────┐
                       server.py                  client.py
                    (loads model.pkl)        (connects to server)
                    (implements Predict)     (calls Predict)
                    (listens on :5000)       (prints response)
```

The `.proto` file is the only shared dependency between server and client. As long as both use the same `.proto`, they can be written in different languages — Python server, Go client, Java client, etc.

---

## Quick Reference

### Generate stubs

```bash
python -m grpc_tools.protoc -I. --python_out=. --grpc_python_out=. prediction.proto
```

### Minimal server

```python
import grpc
from concurrent import futures
import prediction_pb2_grpc

server = grpc.server(futures.ThreadPoolExecutor(max_workers=10))
prediction_pb2_grpc.add_MLModelServicer_to_server(MyServicer(), server)
server.add_insecure_port("[::]:5000")
server.start()
server.wait_for_termination()
```

### Minimal client

```python
import grpc
import prediction_pb2, prediction_pb2_grpc

with grpc.insecure_channel("localhost:5000") as channel:
    stub = prediction_pb2_grpc.MLModelStub(channel)
    response = stub.Predict(prediction_pb2.PredictRequest(features=[3.5]))
    print(response.prediction)
```

### Proto field number rule

```
NEVER change a field number after publishing.
NEVER reuse a deleted field number.
You MAY rename fields safely.
```

---

## Environment Note

This demo has its own `pyproject.toml` and runs in an **isolated `.venv` inside
`API_gRPC/`**, separate from the root DDODS environment. This is intentional and
must not be merged.

**Why:** `grpcio-tools 1.75` requires `protobuf >= 6.31, < 7`. The root DDODS
env pins `protobuf >= 4.24, < 5` — a hard constraint imposed by `feast`. These
two version ranges do not overlap; no single environment can satisfy both.

```
grpcio-tools 1.75  →  protobuf >= 6.31   ← needs protobuf 6.x
feast >= 0.58      →  protobuf >= 4.24, < 5  ← needs protobuf 4.x
                         INCOMPATIBLE — no solution exists
```

All other lessons in this module (TorchScript, LibTorch, Compression) use the
root env because they do not pull a conflicting protobuf version. The gRPC demo
is the exception.

**Always run from inside `API_gRPC/`:**

```bash
cd API_gRPC
uv sync          # creates API_gRPC/.venv with protobuf 6.x
uv run train.py
uv run server.py
```

Running `uv run` from the repo root would use the root env (protobuf 4.x) and
fail when importing the generated `prediction_pb2.py` stubs.

---

## Official Documentation

- gRPC Python quickstart: https://grpc.io/docs/languages/python/quickstart/
- Protocol Buffers language guide (proto3): https://protobuf.dev/programming-guides/proto3/
- gRPC status codes: https://grpc.github.io/grpc/core/md_doc_statuscodes.html
- grpcurl (CLI tool for inspecting gRPC servers): https://github.com/fullstorydev/grpcurl
