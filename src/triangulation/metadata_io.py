from __future__ import annotations

import json
from pathlib import Path


def load_metadata(json_path: str | Path) -> dict:
    """
    Lädt Simulations-/Mess-Metadaten aus einer JSON-Datei.
    """
    json_path = Path(json_path)

    if not json_path.exists():
        raise FileNotFoundError(f"Metadata-JSON nicht gefunden: {json_path}")

    with open(json_path, "r", encoding="utf-8") as f:
        return json.load(f)