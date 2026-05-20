from __future__ import annotations

from pathlib import Path
import csv

import numpy as np


RAY_FIELDS = [
    "frame_idx",
    "origin_x",
    "origin_y",
    "origin_z",
    "direction_x",
    "direction_y",
    "direction_z",
    "u",
    "v",
]


def save_rays_csv(
    rays: list[dict],
    output_path: str | Path,
) -> Path:
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=RAY_FIELDS)
        writer.writeheader()

        for ray in rays:
            writer.writerow(
                {
                    "frame_idx": int(ray["frame_idx"]),
                    "origin_x": float(ray["origin"][0]),
                    "origin_y": float(ray["origin"][1]),
                    "origin_z": float(ray["origin"][2]),
                    "direction_x": float(ray["direction"][0]),
                    "direction_y": float(ray["direction"][1]),
                    "direction_z": float(ray["direction"][2]),
                    "u": "" if ray.get("uv") is None else float(ray["uv"][0]),
                    "v": "" if ray.get("uv") is None else float(ray["uv"][1]),
                }
            )

    print(f"💾 Rays CSV gespeichert: {output_path}")
    return output_path


def rays_to_array(rays: list[dict]) -> np.ndarray:
    """
    Format:
        [frame_idx,
         origin_x, origin_y, origin_z,
         direction_x, direction_y, direction_z,
         u, v]
    """
    rows = []

    for ray in rays:
        uv = ray.get("uv")

        if uv is None:
            u = np.nan
            v = np.nan
        else:
            u = float(uv[0])
            v = float(uv[1])

        rows.append(
            [
                int(ray["frame_idx"]),
                float(ray["origin"][0]),
                float(ray["origin"][1]),
                float(ray["origin"][2]),
                float(ray["direction"][0]),
                float(ray["direction"][1]),
                float(ray["direction"][2]),
                u,
                v,
            ]
        )

    if len(rows) == 0:
        return np.empty((0, 9), dtype=np.float64)

    return np.asarray(rows, dtype=np.float64)


def save_rays_npy(
    rays: list[dict],
    output_path: str | Path,
) -> Path:
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    array = rays_to_array(rays)
    np.save(output_path, array)

    print(f"💾 Rays NPY gespeichert: {output_path}")
    return output_path


def save_rays(
    rays: list[dict],
    output_dir: str | Path,
    stem: str,
) -> dict:
    """
    Speichert Rays als CSV und NPY.

    Beispiel:
        save_rays(camera_rays, output_dir, "camera_rays")
        -> camera_rays.csv
        -> camera_rays.npy
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    csv_path = save_rays_csv(
        rays=rays,
        output_path=output_dir / f"{stem}.csv",
    )

    npy_path = save_rays_npy(
        rays=rays,
        output_path=output_dir / f"{stem}.npy",
    )

    return {
        "csv": csv_path,
        "npy": npy_path,
    }