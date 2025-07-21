from loguru import logger
from pydantic_settings import BaseSettings

from inference_torch import EngineParams
from log_config import LoggingParams
from models import ModelParams


class AppConfig(BaseSettings):
    model: ModelParams = ModelParams()
    engine: EngineParams = EngineParams()
    logging: LoggingParams = LoggingParams()

    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"

config = AppConfig()
