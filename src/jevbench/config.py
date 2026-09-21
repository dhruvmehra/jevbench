from __future__ import annotations

import os
import tomllib
from pathlib import Path


def load_config(path: str = "config.toml") -> dict:
    return tomllib.loads(Path(path).read_text())


def api_key() -> str:
    from dotenv import load_dotenv

    load_dotenv()
    key = os.environ.get("OPENROUTER_API_KEY")
    if not key:
        raise RuntimeError("OPENROUTER_API_KEY not set (put it in .env or export it)")
    return key
