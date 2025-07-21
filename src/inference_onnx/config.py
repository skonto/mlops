from typing import Literal

import onnxruntime as ort
from pydantic_settings import BaseSettings


class EngineParams(BaseSettings):
    device: Literal["cpu", "cuda"] = "cuda"
    num_threads: int = 8

    @property
    def providers(self):
        if self.device == "cuda":
            if "CUDAExecutionProvider" in ort.get_available_providers():
                return ["CUDAExecutionProvider"]
        return ["CPUExecutionProvider"]

    class Config:
        env_prefix = "ENGINE_"
