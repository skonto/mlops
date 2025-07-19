from pydantic_settings import BaseSettings
from typing import Literal
import torch
from loguru import logger

class EngineParams(BaseSettings):
    device: Literal["cpu", "cuda"] = "cuda"
    num_threads: int = 8
    matmul_precision: Literal["high", "medium", "default"] = "high"

    @property
    def torch_device(self) -> torch.device:
        if self.device == "cuda":
            if torch.cuda.is_available():
                return torch.device("cuda")
            else:
                logger.warning("CUDA requested but not available. Falling back to CPU.")
                return torch.device("cpu")
        return torch.device("cpu")
    
    def apply_runtime_settings(self):
        if hasattr(torch, "set_float32_matmul_precision"):
            torch.set_float32_matmul_precision(self.matmul_precision)
            logger.info(f"Set torch matmul precision to '{self.matmul_precision}'")
        else:
            logger.warning("torch.set_float32_matmul_precision is not available in this PyTorch version.")

    class Config:
        env_prefix = "ENGINE_"