"""Artifact store for large Action payloads."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

from schemas import Artifact


ROOT = Path(__file__).parent
STATE_DIR = ROOT / "state"
ARTIFACT_DIR = STATE_DIR / "artifacts"
INDEX_PATH = ARTIFACT_DIR / "index.json"


def put(
    data: bytes,
    *,
    source: str,
    content_type: str = "text/plain",
    descriptor: str = "",
) -> Artifact:
    ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)
    h = hashlib.sha256(data).hexdigest()
    artifact_id = f"art:{h[:16]}"
    path = _path_for(artifact_id)
    path.write_bytes(data)

    artifact = Artifact(
        id=artifact_id,
        content_type=content_type,
        size_bytes=len(data),
        source=source,
        descriptor=descriptor,
    )
    index = _load_index()
    index[artifact_id] = artifact.model_dump(mode="json")
    INDEX_PATH.write_text(json.dumps(index, indent=2), encoding="utf-8")
    return artifact


def exists(artifact_id: str) -> bool:
    return _path_for(artifact_id).exists()


def get_bytes(artifact_id: str) -> bytes:
    return _path_for(artifact_id).read_bytes()


def _path_for(artifact_id: str) -> Path:
    safe = artifact_id.replace(":", "_")
    return ARTIFACT_DIR / f"{safe}.bin"


def _load_index() -> dict:
    ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)
    if not INDEX_PATH.exists():
        return {}
    return json.loads(INDEX_PATH.read_text(encoding="utf-8") or "{}")
