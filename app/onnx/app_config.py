from pydantic_settings import BaseSettings

from inference_onnx import EngineParams
from log_config import LoggingParams


class AppConfig(BaseSettings):
    engine: EngineParams = EngineParams()
    logging: LoggingParams = LoggingParams()

    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"

config = AppConfig()
