from fastapi import FastAPI
import torch
from models import IrisDL
from data import BatchInput
from inference_torch import TorchInferenceEngine
from contextlib import asynccontextmanager
from pynvml import *
from fastapi import FastAPI, Depends, HTTPException, Request
from utils import start_gpu_monitor, stop_gpu_monitor, report_actual_gpu_memory

config = {
    "input_dim": 4,
    "hidden_dims": [16, 128, 32, 64],
    "output_dim": 3,
    "median": torch.zeros(4),
    "iqr": torch.ones(4),
    "activation": "relu",
    "dropout": 0.1956,
}

@asynccontextmanager
async def lifespan(app: FastAPI):
    global engine
    global device 
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.set_float32_matmul_precision('high')
    model = IrisDL.from_checkpoint("model.pt", config, device)
    engine = TorchInferenceEngine(model=model, device=device, num_threads=4)
    start_gpu_monitor(device = device)
    print("✅ Model and inference engine ready")
    yield
    engine = None
    stop_gpu_monitor()
    
app = FastAPI(lifespan=lifespan)

@app.post("/predict")
async def predict(data: BatchInput):
    return await engine.predict_async(data.features)

@app.get("/healthz")
def health_check():
    return {"status": "ok"}

@app.get("/ready")
def readiness_check():
    if engine is not None:
        return {"ready": True}
    return {"ready": False}

def require_localhost(request: Request) -> Request:
    client_host = request.client.host
    if client_host not in ("127.0.0.1", "::1"):
        raise HTTPException(status_code=403, detail=f"Access denied: not localhost ({client_host})")
    return request  # you can return the request if the handler needs it

@app.get("/debug/gpu", include_in_schema=False, dependencies=[Depends(require_localhost)])
async def get_gpu_memory(request: Request):
    print("CUDA available:", torch.cuda.is_available())
    print("CUDA device count:", torch.cuda.device_count())
    print("Current device:", torch.cuda.current_device())
    print("Device name:", torch.cuda.get_device_name(torch.cuda.current_device()))
    report_actual_gpu_memory(device=device)
    return {"status": "checked"}