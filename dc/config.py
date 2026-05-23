from pathlib import Path

from loguru import logger

PROJ_ROOT = Path(__file__).resolve().parents[1]
logger.info(f"PROJ_ROOT path is: {PROJ_ROOT}")

CONFIG_DIR = PROJ_ROOT / "config"
DATA_DIR = PROJ_ROOT / "data"
OUTPUT_DIR = PROJ_ROOT / "output"
