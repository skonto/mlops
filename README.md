# MLOPS
Examples of ML lifecycle

## Train a model

Here we train a NN for the Iris dataset. Optuna is used to find the optimal architecture and then the trained model
is exported to different formats. For the torch model we compile as we expect it to run on the same card locally.

## Build the container for the model with a custom inference engine

Here we build one container per exported type for the same model.

```
docker build -f deployment/Dockerfile.torch -t fastapi-inference-torch .

docker build -f deployment/Dockerfile.onnx -t fastapi-inference-onnx .

```

## Run the container

Here we run on gpu the corresponding containers. The custom infernece for the torch model also
prints gpu allocations.

```
docker run --gpus all -p8000:8000 fastapi-inference-torch
INFO:     Started server process [1]
INFO:     Waiting for application startup.
INFO:     Application startup complete.
INFO:     Uvicorn running on http://0.0.0.0:8000 (Press CTRL+C to quit)
```


```docker run --gpus all -p8000:8000 fastapi-inference-onnx

==========
== CUDA ==
==========

CUDA Version 12.8.0

Container image Copyright (c) 2016-2023, NVIDIA CORPORATION & AFFILIATES. All rights reserved.

This container image and its contents are governed by the NVIDIA Deep Learning Container License.
By pulling and using the container, you accept the terms and conditions of this license:
https://developer.nvidia.com/ngc/nvidia-deep-learning-container-license

A copy of this license is made available in this container at /NGC-DL-CONTAINER-LICENSE for your convenience.

INFO:     Started server process [1]
INFO:     Waiting for application startup.
INFO:     Application startup complete.
INFO:     Uvicorn running on http://0.0.0.0:8000 (Press CTRL+C to quit)

```

Make predictions:
```
curl -X POST http://localhost:8000/predict -H "Content-Type: application/json"   -d '{"features": [[52.1, 32.4, 22, 1]]}'
{"predictions":[1]}
```

# Deploying with Triton Server

Triton supports multiple formats. Here we will deploy the onnx model.

```
docker run --rm --gpus all -p8000:8000 -p8001:8001 -p8002:8002   -v $PWD/triton_model_repository:/models   nvcr.io/nvidia/tritonserver:25.06-py3   tritonserver --model-repository=/models
...
I0708 15:56:37.437140 1 grpc_server.cc:2562] "Started GRPCInferenceService at 0.0.0.0:8001"
I0708 15:56:37.437510 1 http_server.cc:4832] "Started HTTPService at 0.0.0.0:8000"
I0708 15:56:37.478611 1 http_server.cc:358] "Started Metrics Service at 0.0.0.0:8002"
...

curl -X POST http://localhost:8000/v2/models/iris/infer -H "Content-Type: application/json"   -H "Accept: application/json"   --data-binary @- <<EOF
{
  "inputs": [
    {
      "name": "input",
      "shape": [1, 4],
      "datatype": "FP32",
      "data": [[52.1, 32.4, 22, 1]]
    }
  ],
  "outputs": [
    {
      "name": "output"
    }
  ]
}
EOF
{"model_name":"iris","model_version":"1","outputs":[{"name":"output","datatype":"FP32","shape":[1,3],"data":[-127.28438568115235,188.3376007080078,-269.2401428222656]}]}

```

Here there is no argmax logic but it could be added to the model.

# Deploying with torchserve

```

uv run torch-model-archiver   --model-name iris   --version 1.0 \
--serialized-file model.pt   --handler src/models/torch_handler.py \
--export-path torch_serve --extra-files "src/models/iris.py,src/models/torch_model_config.py,torch_serve/config.properties"

docker build -f deployment/Dockerfile.torchserve -t inference-torchserve .

docker  run --gpus all -p8000:8000 inference-torchserve
curl -X POST http://localhost:8000/predictions/iris -H "Content-Type: application/json"      -d '[[5.1, 3.5, 1.4, 0.2], [6.2, 3.4, 5.4, 2.3]]'
[
  0,
  2
]
```

Torchserve requires a custom handler. Models can be loaded by a management api as well which is disabled by default.
Authntication is also off in this setup. In addition grpc can be enabled along with https via the config.properties file.