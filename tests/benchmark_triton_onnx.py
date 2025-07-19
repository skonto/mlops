import requests
import json
import time
import numpy as np

TRITON_URL = "http://localhost:8000/v2/models/iris/infer"
NUM_REQUESTS = 1000
BATCH_SIZE = 100

def generate_payload(batch_size: int = 1):
    data = [[52.1, 32.4, 22.0, 1.0] for _ in range(batch_size)]

    payload = {
        "inputs": [
            {
                "name": "input",
                "shape": [batch_size, 4],
                "datatype": "FP32",
                "data": data,
            }
        ],
        "outputs": [{"name": "output"}]
    }

    return json.dumps(payload)

headers = {
    "Content-Type": "application/json",
    "Accept": "application/json"
}

latencies = []

for i in range(NUM_REQUESTS):
    payload = generate_payload(BATCH_SIZE)
    start = time.perf_counter()
    response = requests.post(TRITON_URL, headers=headers, data=payload)
    latency = (time.perf_counter() - start) * 1000  # ms
    latencies.append(latency)

    if response.status_code != 200:
        print(f"Request {i} failed: {response.status_code} - {response.text}")

latencies = np.array(latencies)

print(f"\nTriton Inference Benchmark ({NUM_REQUESTS} requests @ batch={BATCH_SIZE}):")
print(f"  Mean latency       : {np.mean(latencies):.2f} ms")
print(f"  Median latency     : {np.percentile(latencies, 50):.2f} ms")
print(f"  P95 latency        : {np.percentile(latencies, 95):.2f} ms")
print(f"  P99 latency        : {np.percentile(latencies, 99):.2f} ms")
print(f"  Max latency        : {np.max(latencies):.2f} ms")
print(f"  Throughput         : {NUM_REQUESTS / np.sum(latencies / 1000):.2f} req/s")
