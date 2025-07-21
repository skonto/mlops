from typing import List, Literal

import torch
from pydantic_settings import BaseSettings


class ModelParams(BaseSettings):
    input_dim: int = 4
    hidden_dims: List[int] = [16, 128, 32, 64]
    output_dim: int = 3
    median_values: List[float] = [0.0, 0.0, 0.0, 0.0]
    iqr_values: List[float] = [1.0, 1.0, 1.0, 1.0]
    activation: Literal["relu", "gelu", "tanh", "sigmoid"] = "relu"
    dropout: float = 0.1956

    def to_model_dict(self) -> dict:
        return {
            "input_dim": self.input_dim,
            "hidden_dims": self.hidden_dims,
            "output_dim": self.output_dim,
            "activation": self.activation,
            "dropout": self.dropout,
            "median": self.median,
            "iqr": self.iqr,
        }

    @property
    def median(self) -> torch.Tensor:
        return torch.tensor(self.median_values)

    @property
    def iqr(self) -> torch.Tensor:
        return torch.tensor(self.iqr_values)

    class Config:
        env_prefix = "MODEL_"


class TrainModelParams(BaseSettings):
    lr: float = 0.0073
    model: ModelParams = ModelParams()

    def to_model_dict(self) -> dict:
        return {
            "input_dim": self.model.input_dim,
            "hidden_dims": self.model.hidden_dims,
            "output_dim": self.model.output_dim,
            "activation": self.model.activation,
            "dropout": self.model.dropout,
            "median": self.model.median,
            "iqr": self.model.iqr,
            "lr": self.lr,
        }

    class Config:
        env_prefix = "TRAIN_MODEL_"
