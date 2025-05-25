from fastapi import FastAPI
from data import BatchInput
from inference import ONNXInferenceEngine
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