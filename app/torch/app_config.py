from pydantic_settings import BaseSettings
from loguru import logger
from log_config import LoggingParams
from inference_torch import EngineParams
from models import ModelParams

class AppConfig(BaseSettings):
    model: ModelParams = ModelParams()
    engine: EngineParams = EngineParams()
    logging: LoggingParams = LoggingParams()

    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"

config = AppConfig()
