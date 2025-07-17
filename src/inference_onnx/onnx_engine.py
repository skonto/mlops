import onnxruntime as ort
import numpy as np
from concurrent.futures import ThreadPoolExecutor
import asyncio
from typing import List, Dict, Any

class ONNXInferenceEngine:
    def __init__(self, onnx_path: str, providers: List[str], num_threads: int = 4):
        self.session = ort.InferenceSession(onnx_path, providers=providers)
        self.input_name = self.session.get_inputs()[0].name
        self.output_name = self.session.get_outputs()[0].name
        self.pool = ThreadPoolExecutor(max_workers=num_threads)

    def predict(self, batch: List[List[float]]) -> Dict[str, Any]:
        x = np.array(batch, dtype=np.float32)
        outputs = self.session.run([self.output_name], {self.input_name: x})
        preds = np.argmax(outputs[0], axis=1).tolist()
        return {"predictions": preds}

    async def predict_async(self, batch: List[List[float]]) -> Dict[str, Any]:
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(self.pool, self.predict, batch)
