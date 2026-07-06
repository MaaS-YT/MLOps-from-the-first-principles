"""
gRPC ML Serving — client side.

This is the third step in the pipeline.  It:
  1. Opens a gRPC channel to the running server
  2. Creates a stub — the client-side proxy for the MLModel service
  3. Calls Predict() for several test inputs and prints the responses

How the client side works:
  - grpc.insecure_channel() establishes an HTTP/2 connection to the server.
    The channel is lazy: it does not actually connect until the first RPC call.
  - The stub wraps the channel and exposes the same methods defined in the
    .proto service block.  Calling stub.Predict(...) serialises the request to
    Protobuf binary, sends it over the channel, waits for the response, and
    deserialises it — all transparently.
  - Using the channel as a context manager (with grpc.insecure_channel(...))
    ensures it is closed and resources are released when the block exits.

The server must be running before this script is executed.

Run with:
    uv run client.py
"""

import grpc
import prediction_pb2
import prediction_pb2_grpc


# ---------------------------------------------------------------------------
# Prediction helper
#
# Wraps a single RPC call.  Taking the stub as a parameter (instead of
# creating it inside the function) lets the caller reuse one channel for
# all requests — opening a new TCP + HTTP/2 connection per call would be
# wasteful.
# ---------------------------------------------------------------------------

def predict(stub: prediction_pb2_grpc.MLModelStub, features: list[float]):
    # Build a PredictRequest Protobuf message.  The `features` field is
    # declared as `repeated float` in the .proto file, so it accepts any
    # iterable of floats.
    request  = prediction_pb2.PredictRequest(features=features)
    response = stub.Predict(request)
    return response


# ---------------------------------------------------------------------------
# Main — send four test inputs and print the structured responses
#
# Expected output for y = 2x:
#   Input  1.0 → prediction  2.00
#   Input  3.5 → prediction  7.00
#   Input  5.0 → prediction 10.00
#   Input 10.0 → prediction 20.00
# ---------------------------------------------------------------------------

def main() -> None:
    test_cases = [[1.0], [3.5], [5.0], [10.0]]

    print("Connecting to gRPC server at localhost:5000...\n")

    # One channel for all requests — channels are thread-safe and reusable.
    with grpc.insecure_channel("localhost:5000") as channel:
        stub = prediction_pb2_grpc.MLModelStub(channel)

        for features in test_cases:
            try:
                response = predict(stub, features)
                print(f"Input      : {features[0]:.1f}")
                print(f"Prediction : {response.prediction:.2f}")
                print(f"Model ver  : {response.model_version}")
                print("-" * 30)
            except grpc.RpcError as e:
                # RpcError carries a structured status code and detail string
                # set by the server's context.set_code / context.set_details.
                print(f"RPC failed [{e.code()}]: {e.details()}")


if __name__ == "__main__":
    main()
