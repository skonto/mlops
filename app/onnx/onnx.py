from contextlib import asynccontextmanager

from fastapi import FastAPI
from loguru import logger
from pynvml import *

import app.onnx.app_config as app_config
from data import BatchInput
from inference_onnx import ONNXInferenceEngine
from log_config import setup_logging


@asynccontextmanager
async def lifespan(app: FastAPI):
    global engine
    engine = ONNXInferenceEngine(onnx_path="model.onnx", providers=app_config.config.engine.providers, num_threads=app_config.config.engine.num_threads)
    logger.info("Model and inference engine ready")
    yield
    engine = None

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
