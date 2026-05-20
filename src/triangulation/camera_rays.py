from __future__ import annotations

import numpy as np


def get_camera_ray_from_pixel(
    u: float,
    v: float,
    metadata: dict,
) -> np.ndarray:
    """
    Wandelt einen Pixelpunkt (u, v) in einen normierten Kamerastrahl um.

    Aktuelles Modell:
    - einfaches Pinhole-Modell
    - cx = img_width / 2
    - cy = img_height / 2
    - keine Verzerrung

    Diese Funktion bleibt bewusst kompatibel zur bisherigen Simulation.
    """
    camera = metadata["camera"]

    img_width = camera["img_width"]
    print("image_widht:", img_width)
    img_height = camera["img_height"]
    focal_length = camera["focal_length"]
    pixel_size = camera["pixel_size"]

    cx = img_width / 2.0
    cy = img_height / 2.0

    x_img = (u - cx) * pixel_size
    y_img = -(v - cy) * pixel_size

    direction = np.array([x_img, y_img, -focal_length], dtype=float)
    return direction / np.linalg.norm(direction)