from __future__ import annotations

import json
from pathlib import Path
import numpy as np


def get_camera_intrinsics_path_for_run(run_folder: str | Path) -> Path:
    return Path(run_folder) / "camera_calibration" / "camera_intrinsics_result.json"


def load_camera_intrinsics_for_run(
    run_folder: str | Path,
    fallback_intrinsics: dict,
) -> dict:
    """
    Lädt Intrinsics aus:
        data/MESSREIHE/camera_calibration/camera_intrinsics_result.json

    Falls nicht vorhanden:
        nutzt fallback_intrinsics aus robot_to_camera_calibration["metadata"]["intrinsics"]
    """
    path = get_camera_intrinsics_path_for_run(run_folder)

    if path.exists():
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)

        K = np.asarray(data["camera_matrix"], dtype=float).reshape(3, 3)
        dist_coeffs = np.asarray(data.get("dist_coeffs", []), dtype=float).reshape(-1)

        image_size = data.get("image_size", {})

        return {
            "source": "camera_intrinsics_result.json",
            "path": path,
            "fx": float(K[0, 0]),
            "fy": float(K[1, 1]),
            "cx": float(K[0, 2]),
            "cy": float(K[1, 2]),
            "img_width": int(image_size.get("width", fallback_intrinsics["img_width"])),
            "img_height": int(image_size.get("height", fallback_intrinsics["img_height"])),
            "dist_coeffs": dist_coeffs,
        }

    return {
        "source": "fallback_robot_to_camera_metadata",
        "path": None,
        "fx": float(fallback_intrinsics["fx"]),
        "fy": float(fallback_intrinsics.get("fy", fallback_intrinsics["fx"])),
        "cx": float(fallback_intrinsics.get("cx", fallback_intrinsics["img_width"] / 2.0)),
        "cy": float(fallback_intrinsics.get("cy", fallback_intrinsics["img_height"] / 2.0)),
        "img_width": int(fallback_intrinsics["img_width"]),
        "img_height": int(fallback_intrinsics["img_height"]),
        "dist_coeffs": np.asarray(fallback_intrinsics.get("dist_coeffs", []), dtype=float),
    }


def _has_distortion(dist_coeffs: np.ndarray) -> bool:
    dist_coeffs = np.asarray(dist_coeffs, dtype=float).reshape(-1)
    return len(dist_coeffs) > 0 and bool(np.any(np.abs(dist_coeffs) > 1e-15))


def camera_ray_from_pixel_with_intrinsics(
    u: float,
    v: float,
    intrinsics: dict,
) -> np.ndarray:
    """
    Baut Kameraray im Kamera-KS.

    Konvention wie bisher:
        +x nach rechts
        +y nach oben
        -z Blickrichtung
    """
    fx = float(intrinsics["fx"])
    fy = float(intrinsics["fy"])
    cx = float(intrinsics["cx"])
    cy = float(intrinsics["cy"])

    dist_coeffs = np.asarray(intrinsics.get("dist_coeffs", []), dtype=float).reshape(-1)

    if _has_distortion(dist_coeffs):
        import cv2

        K = np.array(
            [
                [fx, 0.0, cx],
                [0.0, fy, cy],
                [0.0, 0.0, 1.0],
            ],
            dtype=float,
        )

        uv = np.array([[[float(u), float(v)]]], dtype=float)

        undistorted = cv2.undistortPoints(
            src=uv,
            cameraMatrix=K,
            distCoeffs=dist_coeffs,
            P=None,
        )

        x = float(undistorted[0, 0, 0])
        y = -float(undistorted[0, 0, 1])

    else:
        x = (float(u) - cx) / fx
        y = (cy - float(v)) / fy

    direction = np.array([x, y, -1.0], dtype=float)
    direction /= np.linalg.norm(direction)

    return direction