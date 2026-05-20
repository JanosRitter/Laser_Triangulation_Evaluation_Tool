from __future__ import annotations

import json
from pathlib import Path

import numpy as np


def load_metadata(json_path: str | Path) -> dict:
    json_path = Path(json_path)

    if not json_path.exists():
        raise FileNotFoundError(f"Metadata-JSON nicht gefunden: {json_path}")

    with open(json_path, "r", encoding="utf-8") as f:
        return json.load(f)


def get_project_root() -> Path:
    return Path(__file__).resolve().parents[2]


def get_robot_to_camera_transform_path_for_run(
    run_folder: str | Path,
) -> Path:
    run_folder = Path(run_folder)
    return run_folder / "laser_calibration" / "robot_to_camera_transform.json"


def load_robot_to_camera_calibration(
    run_folder: str | Path | None = None,
    calibration_path: str | Path | None = None,
) -> dict:
    if calibration_path is None:
        if run_folder is None:
            raise ValueError(
                "Entweder run_folder oder calibration_path muss angegeben werden."
            )

        calibration_path = get_robot_to_camera_transform_path_for_run(run_folder)

    calibration_path = Path(calibration_path)

    if not calibration_path.exists():
        raise FileNotFoundError(
            f"robot_to_camera_transform.json nicht gefunden: {calibration_path}"
        )

    with open(calibration_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    if "matrix_T_C_R" not in data:
        raise KeyError("matrix_T_C_R fehlt in robot_to_camera_transform.json")

    T_C_R = np.asarray(data["matrix_T_C_R"], dtype=float)

    if T_C_R.shape != (4, 4):
        raise ValueError(
            f"matrix_T_C_R muss Shape (4,4) haben, ist aber {T_C_R.shape}"
        )

    data["_calibration_path"] = calibration_path
    data["_T_C_R_array"] = T_C_R

    return data