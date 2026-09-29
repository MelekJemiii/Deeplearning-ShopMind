"""Loads settings.yaml + .env once, so no script hardcodes paths, keys or parameters."""
import os
from pathlib import Path

import yaml
from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parents[2]
load_dotenv(ROOT / ".env")

with open(ROOT / "config" / "settings.yaml", encoding="utf-8") as f:
    SETTINGS = yaml.safe_load(f)


def path(key: str) -> Path:
    """Absolute path from settings.paths, relative to the project root."""
    return ROOT / SETTINGS["paths"][key]


def env(key: str, required: bool = True) -> str:
    value = os.getenv(key, "")
    if required and not value:
        raise RuntimeError(f"Missing {key} in .env (see .env.example)")
    return value
