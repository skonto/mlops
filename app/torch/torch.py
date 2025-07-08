from fastapi import FastAPI
import torch
from typing import List
from models import IrisDL
from data import BatchInput
from inference_torch import TorchInferenceEngine
from contextlib import asynccontextmanager
import threading
import time
from typing import Optional
from pynvml import *
import pynvml

best_params = {
    'n_layers': 4,
    'hidden_dim_0': 16,
    'hidden_dim_1': 128,
    'hidden_dim_2': 32,
    'hidden_dim_3': 64,
    'activation': 'relu',
    'dropout': 0.1956,
    'lr': 0.0073
}

hidden_dims = [best_params[f"hidden_dim_{i}"] for i in range(best_params["n_layers"])]

model = None
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("CUDA available:", torch.cuda.is_available())
print("CUDA device count:", torch.cuda.device_count())
print("Current device:", torch.cuda.current_device())
print("Device name:", torch.cuda.get_device_name(torch.cuda.current_device()))
monitor_thread: Optional[threading.Thread] = None
stop_signal = threading.Event()

def report_actual_gpu_memory():
    pynvml.nvmlInit()
    handle = pynvml.nvmlDeviceGetHandleByIndex(0)
    info = nvmlDeviceGetComputeRunningProcesses(handle)
    for p in info:
        print(f"PID {p.pid}: {p.usedGpuMemory / 1024**2:.2f} MB") # same number as nvidia-smi
    pynvml.nvmlShutdown()

def gpu_monitor_loop(interval: float = 5.0, device: torch.device = torch.device("cuda:0")):
    while not stop_signal.is_set():
        if torch.cuda.is_available():
            allocated = torch.cuda.memory_allocated() / 1024**2
            reserved = torch.cuda.memory_reserved() / 1024**2
            print(f"[GPU] Allocated: {allocated:.2f} MB")
            print(f"[GPU] Reserved: {reserved:.2f} MB")
            report_actual_gpu_memory()

            time.sleep(interval)

@asynccontextmanager
async def lifespan(app: FastAPI):
    global engine
    global monitor_thread
    print("✅ Starting GPU memory monitor...")
    stop_signal.clear()
    monitor_thread = threading.Thread(target=gpu_monitor_loop, daemon=True)
    monitor_thread.start()
    dummy = torch.zeros(4)
    model = IrisDL(
    input_dim=4,
    hidden_dims=hidden_dims,
    output_dim=3,
    median= dummy,
    iqr= dummy,
    activation=best_params["activation"],
    dropout=best_params["dropout"])
    torch.set_float32_matmul_precision('high')
    torch.load("compiled_model.pt", map_location=device, weights_only=False)
    engine = TorchInferenceEngine(model=model, device=device, num_threads=4)
    print("✅ Model and inference engine ready")
    
    yield
    engine = None
    print("🛑 Inference engine shut down")
    print("🛑 Stopping GPU memory monitor...")
    stop_signal.set()
    monitor_thread.join()

app = FastAPI(lifespan=lifespan)

@app.post("/predict")
async def predict(data: BatchInput):
    return await engine.predict_async(data.features)