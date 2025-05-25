# MLOPS
Examples of ML lifecycle

## Build the container for the model

Here we build one container per exported type for the same model.

```
docker build -f deployment/Dockerfile.torch -t fastapi-inference .

docker build -f deployment/Dockerfile.onnx -t fastapi-inference-onnx .

```

## Run the container

Here we run on gpu the corresponding containers. The custom infernece for the torch model also
prints gpu allocations.

```
docker run --gpus all -p8000:8000 fastapi-inference
CUDA available: True
CUDA device count: 1
Current device: 0
Device name: NVIDIA GeForce RTX 4060 Laptop GPU
INFO:     Started server process [1]
INFO:     Waiting for application startup.
✅ Starting GPU memory monitor...
[GPU] Allocated: 0.00 MB
[GPU] Reserved: 0.00 MB
INFO:     Application startup complete.
✅ Model and inference engine ready
INFO:     Uvicorn running on http://0.0.0.0:8000 (Press CTRL+C to quit)
```

```docker run --gpus all -p8000:8000 fastapi-inference-onnx
INFO:     Started server process [1]
INFO:     Waiting for application startup.
✅ Model and inference engine ready
INFO:     Application startup complete.
INFO:     Uvicorn running on http://0.0.0.0:8000 (Press CTRL+C to quit)
```
