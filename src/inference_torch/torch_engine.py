import torch
from torch import nn
from concurrent.futures import ThreadPoolExecutor
import asyncio
from typing import List, Dict, Any
import numpy as np

class TorchInferenceEngine:
    def __init__(self, model: nn.Module, device: torch.device, num_threads: int = 4):
        self.model = model.to(device)
        self.device = device
        self.pool = ThreadPoolExecutor(max_workers=num_threads)
        self.model.eval()

    def predict(self, batch: List[List[float]]) -> Dict[str, Any]:
        x = torch.from_numpy(np.asarray(batch, dtype=np.float32)).to(self.device)
        with torch.no_grad():
            out = self.model(x)
            preds = torch.argmax(out, dim=1).tolist()
        return {"predictions": preds}

    async def predict_async(self, batch: List[List[float]]) -> Dict[str, Any]:
        loop = asyncio.get_running_loop()
        return await loop.run_in_executor(self.pool, self.predict, batch)
