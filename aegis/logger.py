import sys
from loguru import logger

logger.remove()
logger.add(sys.stdout, format="<green>{time:YYYY-MM-DD HH:mm:ss}</green> | <level>{level: <8}</level> | <cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> - <level>{message}</level>")
# In production, this would also write to a rotating file or ElasticSearch
logger.add("aegis_mdt.log", rotation="10 MB", retention="10 days", level="INFO")

def get_logger(name: str):
    return logger.bind(module=name)
