import torch
from torch import nn
from concurrent.futures import ThreadPoolExecutor
import asyncio
from typing import List, Dict, Any
import onnxruntime as ort
import numpy as np

class InferenceEngine:
    def __init__(self, model: nn.Module, device: torch.device, num_threads: int = 4):
        self.model = model.to(device)
        self.device = device
        self.pool = ThreadPoolExecutor(max_workers=num_threads)
        self.model.eval()

    def predict(self, batch: List[List[float]]) -> Dict[str, Any]:
        x = torch.tensor(batch, dtype=torch.float32).to(self.device)
        with torch.no_grad():
            out = self.model(x)
            preds = torch.argmax(out, dim=1).tolist()
        return {"predictions": preds}

    async def predict_async(self, batch: List[List[float]]) -> Dict[str, Any]:
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(self.pool, self.predict, batch)


class ONNXInferenceEngine:
    def __init__(self, onnx_path: str, num_threads: int = 4):
        use_cuda = torch.cuda.is_available()
        providers = ["CUDAExecutionProvider"] if use_cuda else ["CPUExecutionProvider"]
        self.session = ort.InferenceSession(onnx_path, providers=providers)
        self.input_name = self.session.get_inputs()[0].name
        self.output_name = self.session.get_outputs()[0].name
        self.pool = ThreadPoolExecutor(max_workers=num_threads)

    def predict(self, batch: List[List[float]]) -> Dict[str, Any]:
        # Convert input to numpy array
        x = np.array(batch, dtype=np.float32)
        # Run ONNX model
        outputs = self.session.run([self.output_name], {self.input_name: x})
        # Get prediction (e.g., argmax if it's classification)
        logits = outputs[0]
        print("ONNX logits:", logits) 
        preds = np.argmax(outputs[0], axis=1).tolist()
        return {"predictions": preds}

    async def predict_async(self, batch: List[List[float]]) -> Dict[str, Any]:
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(self.pool, self.predict, batch)