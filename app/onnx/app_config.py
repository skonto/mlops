from pydantic_settings import BaseSettings
from log_config import LoggingParams
from inference_onnx import EngineParams

class AppConfig(BaseSettings):
    engine: EngineParams = EngineParams()
    logging: LoggingParams = LoggingParams()

    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"

config = AppConfig()
