from __future__ import annotations

import numpy as np


def get_laser_base_direction_from_metadata(metadata: dict) -> np.ndarray:
    """
    Berechnet die Basisrichtung des Lasers aus den gespeicherten Rotationen.
    """
    rx_deg = metadata["laser"]["rotation_x_deg"]
    ry_deg = metadata["laser"]["rotation_y_deg"]

    rx = np.deg2rad(rx_deg)
    ry = np.deg2rad(ry_deg)

    direction = np.array([0.0, 0.0, 1.0])

    Rx = np.array([
        [1, 0, 0],
        [0, np.cos(rx), -np.sin(rx)],
        [0, np.sin(rx),  np.cos(rx)],
    ])

    Ry = np.array([
        [ np.cos(ry), 0, np.sin(ry)],
        [0,           1, 0],
        [-np.sin(ry), 0, np.cos(ry)],
    ])

    base_dir = Ry @ Rx @ direction
    return base_dir / np.linalg.norm(base_dir)


def get_doe_angle_for_index(
    idx: int,
    n: int,
    fov_deg: float,
    has_center_point: bool,
) -> float:
    """
    Berechnet den Winkel eines DOE-Strahls entlang einer Achse.
    """
    if n <= 0:
        raise ValueError("DOE-Achsgröße n muss > 0 sein.")

    if idx == 0:
        return 0.0

    if n == 1:
        return 0.0

    step_deg = fov_deg / (n - 1)

    if n % 2 == 1:
        angle_deg = idx * step_deg
    else:
        angle_deg = np.sign(idx) * (abs(idx) - 0.5) * step_deg

    return np.deg2rad(angle_deg)


def get_doe_direction_from_index(
    idx_x: int,
    idx_y: int,
    metadata: dict,
) -> np.ndarray:
    """
    Berechnet die Richtung eines DOE-Teilstrahls aus seinem DOE-Index.
    """
    doe = metadata["doe"]

    nx = doe["nx"]
    ny = doe["ny"]
    fov_x_deg = doe["fov_x_deg"]
    fov_y_deg = doe["fov_y_deg"]
    center_point = doe["center_point"]

    base_dir = get_laser_base_direction_from_metadata(metadata)

    ax = get_doe_angle_for_index(idx_x, nx, fov_x_deg, center_point)
    ay = get_doe_angle_for_index(idx_y, ny, fov_y_deg, center_point)

    rx = ay
    ry = ax

    Rx = np.array([
        [1, 0, 0],
        [0, np.cos(rx), -np.sin(rx)],
        [0, np.sin(rx),  np.cos(rx)],
    ])

    Ry = np.array([
        [ np.cos(ry), 0, np.sin(ry)],
        [0,           1, 0],
        [-np.sin(ry), 0, np.cos(ry)],
    ])

    dir_vec = Ry @ Rx @ base_dir
    return dir_vec / np.linalg.norm(dir_vec)


def get_doe_laser_ray_from_index(
    idx_x: int,
    idx_y: int,
    metadata: dict,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Gibt Ursprung und Richtung eines DOE-Laserstrahls zurück.
    """
    laser_origin = np.array(metadata["laser"]["position"], dtype=float)
    laser_direction = get_doe_direction_from_index(idx_x, idx_y, metadata)

    return laser_origin, laser_direction