from fastapi import FastAPI
from data import BatchInput
from inference_onnx import ONNXInferenceEngine
from contextlib import asynccontextmanager
from pynvml import *

@asynccontextmanager
async def lifespan(app: FastAPI):
    global engine
    engine = ONNXInferenceEngine("model.onnx")
    print("✅ Model and inference engine ready")
    yield
    engine = None

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