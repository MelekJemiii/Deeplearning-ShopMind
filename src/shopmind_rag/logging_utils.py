"""Logging to console + file, so every pipeline run leaves a trace."""
import logging
from pathlib import Path

from .config import ROOT


def get_logger(name: str) -> logging.Logger:
    logger = logging.getLogger(name)
    if logger.handlers:
        return logger
    logger.setLevel(logging.INFO)
    fmt = logging.Formatter("%(asctime)s | %(levelname)-7s | %(name)s | %(message)s", "%H:%M:%S")
    (ROOT / "logs").mkdir(exist_ok=True)
    for h in (logging.StreamHandler(), logging.FileHandler(ROOT / "logs" / f"{name}.log", encoding="utf-8")):
        h.setFormatter(fmt)
        logger.addHandler(h)
    return logger
