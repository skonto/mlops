from loguru import logger
from pydantic_settings import BaseSettings


def setup_logging(
    log_file: str = "app.log",
    level: str = "DEBUG",
    rotation: str = "10 MB",
    retention: str = "7 days",
    backtrace: bool = True,
    diagnose: bool = True
):
    logger.remove()
    logger.add(
        log_file,
        rotation=rotation,
        retention=retention,
        backtrace=backtrace,
        diagnose=diagnose,
        level=level,
    )
    logger.debug("Logging is configured.")

class LoggingParams(BaseSettings):
    level: str = "DEBUG"
    log_file: str = "app.log"
    rotation: str = "10 MB"
    retention: str = "7 days"

    class Config:
        env_prefix = "LOG_"
