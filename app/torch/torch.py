from contextlib import asynccontextmanager

import torch
from fastapi import Depends, FastAPI, HTTPException, Request
from loguru import logger
from pynvml import *

import app.torch.app_config as app_config
from data import BatchInput
from inference_torch import TorchInferenceEngine
from log_config import setup_logging
from models import IrisDL
from utils import report_actual_gpu_memory, start_gpu_monitor, stop_gpu_monitor


@asynccontextmanager
async def lifespan(app: FastAPI):
    global engine
    global device
    device = app_config.config.engine.torch_device
    app_config.config.engine.apply_runtime_settings()
    model = IrisDL.from_checkpoint("model.pt", app_config.config.model.to_model_dict(), device)
    engine = TorchInferenceEngine(model=model, device=device, num_threads=app_config.config.engine.num_threads)
    start_gpu_monitor(device = device)
    logger.info("Model and inference engine ready")

    yield
    engine = None
    stop_gpu_monitor()

setup_logging(**app_config.config.logging.model_dump())
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
    return request

@app.get("/debug/gpu", include_in_schema=False, dependencies=[Depends(require_localhost)])
async def get_gpu_memory(request: Request):
    logger.info("CUDA available:", torch.cuda.is_available())
    logger.info("CUDA device count:", torch.cuda.device_count())
    logger.info("Current device:", torch.cuda.current_device())
    logger.info("Device name:", torch.cuda.get_device_name(torch.cuda.current_device()))
    report_actual_gpu_memory(device=device)
    return {"status": "checked"}