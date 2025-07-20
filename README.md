# 🌱 MLOps Examples  
**End-to-end ML pipelines — from training to serving across multiple frameworks**

![License](https://img.shields.io/github/license/skonto/mlops)
![Python](https://img.shields.io/badge/python-3.13%2B-blue)
![Docker](https://img.shields.io/badge/docker-ready-blue)

---

## 🎯 Project Goal & MLOps Vision

This project aims to demonstrate progressive MLOps maturity, inspired by [Google’s MLOps maturity model](https://cloud.google.com/architecture/mlops-continuous-delivery-and-automation-pipelines-in-machine-learning).

### Key Concepts:
- **GitHub as the source of truth**: All code, configuration, and promotion logic lives in GitHub.
- **MLflow** is used to log experiments (see notebooks) and track metadata such as:
  - Git commit hash
  - Model configuration
  - Artifacts for reproducible deployment
- **GitHub Actions** are explored to automate the promotion of models across environments (e.g. from development to staging/production) triggered via PR merges.

### Current Status:
- 🔁 CI/CD is not yet implemented — promotion pipelines are triggered on PR merge.
- 🧪 Not full GitOps — no state reconciliation between deployed models and registry.
- 🕓 Continuous Training (CT) is planned — will enable pipeline re-training on fresh data periodically (e.g. on a schedule).
- 🧩 Feature Store support is not yet implemented (optional for this use case).

> 📝 A future enhancement will show how to use [KitOps](https://kitops.org/docs/get-started) to manage model promotion **without relying on a centralized model registry.**


## 🚀 What’s Inside

Explore fully functional ML deployments of the Iris dataset via different tooling and frameworks:

- **app/** — FastAPI + GPU inference with PyTorch or ONNX  
- **data/** — raw Iris dataset / data ingestion  
- **deployment/** — Dockerfiles, Kubernetes-ready configs  
- **notebooks/** — Jupyter notebooks for training and tuning  
- **src/** — core model definition and training scripts  
- **tests/** — unit/integration tests  
- **torch_serve/** — TorchServe `.mar` archives & handler configs  
- **triton_model_repository/** — Triton Server model repo structure  
- **model.\*** — pre-trained model artifacts (`.pt`, `.onnx`, `.compiled.pt`)  
- **pyproject.toml**, **uv.lock** — dependency management  
- **LICENSE**, **.gitignore**, etc.

---

## 🛠️ Getting Started

1. **Install dependencies**  
   ```bash
   uv sync --frozen --no-editable --no-cache --extra torch --extra torchserve
   ```

2. **Train & export models**  
   ```bash
   uv run src/train/train_iris.py
   ```  
   Outputs:
   - `model.pt` — PyTorch weights  
   - `model.onnx` — export via ONNX  

3. **Build Docker inference containers**

   🚀 **FastAPI + PyTorch**  
   ```bash
   docker build -f Dockerfile.torch -t fastapi-torch .
   docker run --gpus all -p 8000:8000 fastapi-torch
   ```
   Test:
   ```bash
   curl -X POST http://localhost:8000/predict \
     -H "Content-Type: application/json" \
     -d '{"features":[[5.1,3.5,1.4,0.2]]}'
   ```

   🚀 **FastAPI + ONNX**  
   ```bash
   docker build -f Dockerfile.onnx -t fastapi-onnx .
   docker run --gpus all -p 8000:8000 fastapi-onnx
   ```

4. **TorchServe Deployment**

   ```bash
   uv run torch-model-archiver \
     --model-name iris \
     --version 1.0 \
     --serialized-file model.pt \
     --handler src/models/torch_handler.py \
     --export-path torch_serve \
     --extra-files "src/models/iris.py,src/models/torch_model_config.py,torch_serve/config.properties"

   docker build -f deployment/Dockerfile.torchserve -t iris-serve .
   docker run --gpus all -p 8080:8080 iris-serve
   curl -X POST http://localhost:8080/predictions/iris \
     -H "Content-Type: application/json" \
     -d '[[5.1,3.5,1.4,0.2],[6.2,3.4,5.4,2.3]]'
   ```

5. **Triton Inference Server**

   ```bash
   docker run --rm --gpus all \
     -p 8000:8000 -p 8001:8001 -p 8002:8002 \
     -v $PWD/triton_model_repository:/models \
     nvcr.io/nvidia/tritonserver:25.06-py3 \
     tritonserver --model-repository=/models
   ```

   Test:
   ```bash
   curl -X POST http://localhost:8000/v2/models/iris/infer \
     -H "Content-Type: application/json" \
     --data-binary @- <<EOF
   {
     "inputs":[{"name":"input","shape":[1,4],"datatype":"FP32","data":[[5.1,3.5,1.4,0.2]]}],
     "outputs":[{"name":"output"}]
   }
   EOF
   ```

---

## 💡 What You'll Learn

- Model training and tuning with **Optuna + PyTorch**  
- Export to **TorchScript** and **ONNX**  
- GPU-powered inference via **FastAPI**, **TorchServe**, and **Triton**  
- Containerization workflows and best practices  
- How to register and serve models in diverse environments

---

## 📚 Testing & CI Integration

- Run **unit tests**:  
  ```bash
  uv run pytest -q
  ```
- CI/CD integrations can easily be added via GitHub Actions or other platforms, using `uv sync`, `pytest`, and Docker builds.

---

## 🤝 Contributing

We welcome:
- Corrections or bug fixes  
- New deployment workflows (e.g., KServe, TFKeras)  
- Improvements in deployment docs or examples

Please open an issue or submit a pull request!

---

## 📄 License

Apache 2.0 — see **LICENSE** for details.