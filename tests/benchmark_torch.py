import time
import requests
import numpy as np

URL = "http://localhost:8000/predict"  # adjust if different
NUM_REQUESTS = 1000
BATCH_SIZE = 100

# Dummy Iris-like input
def generate_input():
    return {
        "features": [
            [5.1, 3.5, 1.4, 0.2] for _ in range(BATCH_SIZE)
        ]
    }

latencies = []

for i in range(NUM_REQUESTS):
    payload = generate_input()
    start = time.perf_counter()
    response = requests.post(URL, json=payload)
    latency = (time.perf_counter() - start) * 1000  # ms
    latencies.append(latency)

    if response.status_code != 200:
        print(f"Request {i} failed: {response.status_code} - {response.text}")

# Report
latencies = np.array(latencies)
print(f"\nInference Endpoint Benchmark ({NUM_REQUESTS} requests):")
print(f"  Mean latency       : {np.mean(latencies):.2f} ms")
print(f"  P50 latency (median): {np.percentile(latencies, 50):.2f} ms")
print(f"  P95 latency        : {np.percentile(latencies, 95):.2f} ms")
print(f"  P99 latency        : {np.percentile(latencies, 99):.2f} ms")
print(f"  Max latency        : {np.max(latencies):.2f} ms")
print(f"  Throughput         : {NUM_REQUESTS / np.sum(latencies / 1000):.2f} req/s")
