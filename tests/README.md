# Benchmarking Inference

This is a local test and comes with limitations:

```
$ uv run uvicorn app.torch.torch:app --host 0.0.0.0 --port 8000 --workers 1 --no-access-log

# Torch
$ uv run tests/benchmark_torch.py 

Inference Endpoint Benchmark (1000 requests):
  Mean latency       : 1.65 ms
  P50 latency (median): 1.52 ms
  P95 latency        : 1.74 ms
  P99 latency        : 2.11 ms
  Max latency        : 91.94 ms
  Throughput         : 604.54 req/s

# Onnx 
$ uv run tests/benchmark_torch.py 

Inference Endpoint Benchmark (1000 requests):
  Mean latency       : 2.28 ms
  P50 latency (median): 2.20 ms
  P95 latency        : 2.37 ms
  P99 latency        : 2.75 ms
  Max latency        : 50.61 ms
  Throughput         : 439.45 req/s

$ docker run --rm --gpus all -p8000:8000 -p8001:8001 -p8002:8002   -v $PWD/triton_model_repository:/models   nvcr.io/nvidia/tritonserver:25.06-py3   tritonserver --model-repository=/models

$ curl http://localhost:8000/v2/models/iris/config | jq .
{
  "name": "iris",
  "platform": "onnxruntime_onnx",
  "backend": "onnxruntime",
  "runtime": "",
  "version_policy": {
    "latest": {
      "num_versions": 1
    }
  },
  "max_batch_size": 100,
  "input": [
    {
      "name": "input",
      "data_type": "TYPE_FP32",
      "format": "FORMAT_NONE",
      "dims": [
        4
      ],
      "is_shape_tensor": false,
      "allow_ragged_batch": false,
      "optional": false,
      "is_non_linear_format_io": false
    }
  ],
  "output": [
    {
      "name": "output",
      "data_type": "TYPE_FP32",
      "dims": [
        3
      ],
      "label_filename": "",
      "is_shape_tensor": false,
      "is_non_linear_format_io": false
    }
  ],
  "batch_input": [],
  "batch_output": [],
  "optimization": {
    "priority": "PRIORITY_DEFAULT",
    "input_pinned_memory": {
      "enable": true
    },
    "output_pinned_memory": {
      "enable": true
    },
    "gather_kernel_buffer_threshold": 0,
    "eager_batching": false
  },
  "instance_group": [
    {
      "name": "iris",
      "kind": "KIND_GPU",
      "count": 1,
      "gpus": [
        0
      ],
      "secondary_devices": [],
      "profile": [],
      "passive": false,
      "host_policy": ""
    }
  ],
  "default_model_filename": "model.onnx",
  "cc_model_filenames": {},
  "metric_tags": {},
  "parameters": {},
  "model_warmup": []
}

uv run tests/benchmark_triton_onnx.py 

Triton Inference Benchmark (1000 requests @ batch=100):
  Mean latency       : 1.48 ms
  Median latency     : 1.07 ms
  P95 latency        : 1.78 ms
  P99 latency        : 2.09 ms
  Max latency        : 319.01 ms
  Throughput         : 674.51 req/s

```