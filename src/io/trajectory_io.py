from __future__ import annotations

import csv
import json
from pathlib import Path
import numpy as np

from src.utils.path_utils import IMAGE_INPUT_DIR


def resolve_trajectory_input_folder(folder_name_or_path: str | Path) -> Path:
    """
    Löst einen Eingabeordner für trajectory-Daten auf.

    Regeln:
    - absoluter Pfad bleibt absolut
    - relativer Pfad wird relativ zu data/input/images interpretiert
    """
    folder = Path(folder_name_or_path)

    if folder.is_absolute():
        resolved = folder
    else:
        resolved = IMAGE_INPUT_DIR / folder

    if not resolved.exists():
        raise FileNotFoundError(f"Trajectory-Input-Ordner nicht gefunden: {resolved}")

    if not resolved.is_dir():
        raise NotADirectoryError(f"Pfad ist kein Ordner: {resolved}")

    return resolved


def load_run_metadata(input_folder: str | Path) -> dict:
    """
    Lädt run_metadata.json aus einem trajectory-Ordner.
    """
    folder = resolve_trajectory_input_folder(input_folder)
    json_path = folder / "run_metadata.json"

    if not json_path.exists():
        raise FileNotFoundError(f"run_metadata.json nicht gefunden: {json_path}")

    with open(json_path, "r", encoding="utf-8") as f:
        return json.load(f)


def _convert_csv_value(value: str):
    """
    Versucht CSV-Werte sinnvoll zu typisieren.
    """
    if value is None:
        return ""

    value = value.strip()

    if value == "":
        return ""

    if value.lower() == "true":
        return True

    if value.lower() == "false":
        return False

    try:
        if "." in value or "e" in value.lower():
            return float(value)
        return int(value)
    except ValueError:
        return value


def load_frame_table(input_folder: str | Path) -> list[dict]:
    """
    Lädt frame_table.csv als Liste von Dictionaries.
    """
    folder = resolve_trajectory_input_folder(input_folder)
    csv_path = folder / "frame_table.csv"

    if not csv_path.exists():
        raise FileNotFoundError(f"frame_table.csv nicht gefunden: {csv_path}")

    rows = []
    with open(csv_path, "r", newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            converted = {k: _convert_csv_value(v) for k, v in row.items()}
            rows.append(converted)

    return rows


def filter_valid_crop_rows(frame_table: list[dict]) -> list[dict]:
    """
    Filtert valide Frames mit vorhandener Crop-Datei.
    """
    valid_rows = []

    for row in frame_table:
        if row.get("status") != "valid":
            continue

        crop_file = row.get("crop_npy_file", "")
        if not crop_file:
            continue

        valid_rows.append(row)

    return valid_rows


def load_crop_array(input_folder: str | Path, frame_row: dict) -> np.ndarray:
    """
    Lädt das Crop-Array eines Frames anhand von crop_npy_file.
    """
    folder = resolve_trajectory_input_folder(input_folder)

    crop_rel_path = frame_row.get("crop_npy_file", "")
    if not crop_rel_path:
        raise ValueError(
            f"Frame {frame_row.get('frame_idx', '?')} enthält keine crop_npy_file."
        )

    crop_path = folder / crop_rel_path

    if not crop_path.exists():
        raise FileNotFoundError(f"Crop-Datei nicht gefunden: {crop_path}")

    return np.load(crop_path)


def iter_valid_crop_frames(input_folder: str | Path):
    """
    Iterator über alle validen Crop-Frames.

    Yields
    ------
    frame_row : dict
    crop_array : np.ndarray
    """
    frame_table = load_frame_table(input_folder)
    valid_rows = filter_valid_crop_rows(frame_table)

    for row in valid_rows:
        crop_array = load_crop_array(input_folder, row)
        yield row, crop_array