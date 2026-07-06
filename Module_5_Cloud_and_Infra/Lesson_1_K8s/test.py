import requests
import time

data = {"x": 7.5}

start = time.time()
response = requests.post("http://127.0.0.1:8000/predict", json=data)
end = time.time()

print("Response:", response.json())
print(f"Time taken: {end - start:.4f} seconds")
