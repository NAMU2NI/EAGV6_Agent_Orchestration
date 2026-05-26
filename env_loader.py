"""Small .env loader with python-dotenv fallback.

The project still declares python-dotenv as a dependency, but local smoke
checks should not fail just because the raw interpreter lacks that package.
"""
from __future__ import annotations

import os
from pathlib import Path


def load_env(path: str | Path, *, override: bool = False) -> bool:
    """Load KEY=VALUE pairs from a .env file.

    If python-dotenv is available, delegate to it. Otherwise parse the common
    simple form used by this project. Returns True when the file exists.
    """
    try:
        from dotenv import load_dotenv

        return bool(load_dotenv(path, override=override, encoding="utf-8-sig"))
    except ModuleNotFoundError:
        return _load_simple_env(Path(path), override=override)


def _load_simple_env(path: Path, *, override: bool) -> bool:
    if not path.exists():
        return False

    for raw_line in path.read_text(encoding="utf-8-sig").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        if line.startswith("export "):
            line = line[len("export "):].strip()
        if "=" not in line:
            continue
        key, value = line.split("=", 1)
        key = key.strip()
        value = _clean_value(value.strip())
        if not key:
            continue
        if override or key not in os.environ:
            os.environ[key] = value
    return True


def _clean_value(value: str) -> str:
    if len(value) >= 2 and value[0] == value[-1] and value[0] in ("'", '"'):
        return value[1:-1]
    return value
