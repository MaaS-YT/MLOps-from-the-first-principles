import requests
import random
import time
from concurrent.futures import ThreadPoolExecutor, as_completed


def make_prediction_request():
    """Make a single prediction request"""
    data = {
        "age": random.gauss(30, 4),
        "income": random.gauss(70000, 15000),
        "transactions": random.gauss(15, 5)
    }
    try:
        response = requests.post(
            "http://localhost:8000/predict", json=data, timeout=5)
        return response.status_code
    except Exception as e:
        return f"Error: {e}"


# Number of total requests and concurrent workers
total_requests = 5000
max_workers = 20

print(
    f"Sending {total_requests} requests with {max_workers} parallel workers...")
start_time = time.time()

with ThreadPoolExecutor(max_workers=max_workers) as executor:
    futures = [executor.submit(make_prediction_request)
               for _ in range(total_requests)]

    completed = 0
    for future in as_completed(futures):
        completed += 1
        if completed % 100 == 0:
            print(f"Completed {completed}/{total_requests} requests")

elapsed_time = time.time() - start_time
print(
    f"\nFinished! Sent {total_requests} requests in {elapsed_time:.2f} seconds")
print(f"Rate: {total_requests/elapsed_time:.2f} requests/second")

