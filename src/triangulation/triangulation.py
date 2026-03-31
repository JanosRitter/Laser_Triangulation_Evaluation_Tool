import json
from pathlib import Path
import numpy as np


def load_metadata(json_path: str | Path) -> dict:
    """
    Lädt die Simulations-Metadaten aus einer JSON-Datei.
    """
    json_path = Path(json_path)

    if not json_path.exists():
        raise FileNotFoundError(f"Metadata-JSON nicht gefunden: {json_path}")

    with open(json_path, "r", encoding="utf-8") as f:
        return json.load(f)


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
        [0, np.sin(rx),  np.cos(rx)]
    ])

    Ry = np.array([
        [ np.cos(ry), 0, np.sin(ry)],
        [0,           1, 0],
        [-np.sin(ry), 0, np.cos(ry)]
    ])

    base_dir = Ry @ Rx @ direction
    return base_dir / np.linalg.norm(base_dir)


def get_doe_angle_for_index(idx: int, n: int, fov_deg: float, has_center_point: bool) -> float:
    """
    Berechnet den Winkel eines DOE-Strahls entlang einer Achse
    aus seinem zentrierten Index.
    """
    if n <= 0:
        raise ValueError("DOE-Achsengröße n muss > 0 sein.")

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


def get_doe_direction_from_index(idx_x: int, idx_y: int, metadata: dict) -> np.ndarray:
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
        [0, np.sin(rx),  np.cos(rx)]
    ])

    Ry = np.array([
        [ np.cos(ry), 0, np.sin(ry)],
        [0,           1, 0],
        [-np.sin(ry), 0, np.cos(ry)]
    ])

    dir_vec = Ry @ Rx @ base_dir
    return dir_vec / np.linalg.norm(dir_vec)


def get_camera_ray_from_pixel(u: float, v: float, metadata: dict) -> np.ndarray:
    """
    Wandelt einen Pixelpunkt (u,v) in einen normierten Kamerastrahl um.

    Achtung:
    Diese Funktion muss exakt zur in der Simulation verwendeten
    project()-Funktion passen.
    """
    camera = metadata["camera"]

    img_width = camera["img_width"]
    img_height = camera["img_height"]
    focal_length = camera["focal_length"]
    pixel_size = camera["pixel_size"]

    cx = img_width / 2.0
    cy = img_height / 2.0

    x_img = (u - cx) * pixel_size
    y_img = -(v - cy) * pixel_size

    direction = np.array([x_img, y_img, focal_length], dtype=float)
    return direction / np.linalg.norm(direction)


def find_closest_point_between_lines(
    point1: np.ndarray,
    direction1: np.ndarray,
    point2: np.ndarray,
    direction2: np.ndarray
) -> tuple[np.ndarray, float]:
    """
    Berechnet den Mittelpunkt des kürzesten Verbindungsstücks zwischen zwei Geraden.

    Returns
    -------
    midpoint : np.ndarray, shape (3,)
        Rekonstruierter 3D-Punkt
    distance : float
        Minimaler Abstand der beiden Geraden
    """
    d1 = direction1 / np.linalg.norm(direction1)
    d2 = direction2 / np.linalg.norm(direction2)

    diff = point1 - point2
    dot_a = np.dot(d1, d1)
    dot_b = np.dot(d1, d2)
    dot_c = np.dot(d2, d2)
    dot_d = np.dot(d1, diff)
    dot_e = np.dot(d2, diff)

    denom = dot_a * dot_c - dot_b ** 2

    if np.isclose(denom, 0.0):
        raise ValueError("Geraden sind parallel oder numerisch instabil.")

    s = (dot_b * dot_e - dot_c * dot_d) / denom
    t = (dot_a * dot_e - dot_b * dot_d) / denom

    point_on_line1 = point1 + s * d1
    point_on_line2 = point2 + t * d2

    midpoint = 0.5 * (point_on_line1 + point_on_line2)
    distance = np.linalg.norm(point_on_line1 - point_on_line2)

    return midpoint, distance


def triangulate_indexed_points(indexed_points: np.ndarray, metadata: dict) -> np.ndarray:
    """
    Trianguliert DOE-Punkte aus indizierten Bildpunkten.

    Erwartetes Format von indexed_points:
        [idx_x, idx_y, u, v]

    Rückgabeformat:
        [idx_x, idx_y, x, y, z, u, v, line_distance]
    """
    laser_pos = np.array(metadata["laser"]["position"], dtype=float)
    camera_pos = np.array([0.0, 0.0, 0.0], dtype=float)

    results = []

    for row in indexed_points:
        idx_x = int(row[0])
        idx_y = int(row[1])
        u = float(row[2])
        v = float(row[3])

        laser_dir = get_doe_direction_from_index(idx_x, idx_y, metadata)
        cam_dir = get_camera_ray_from_pixel(u, v, metadata)

        point_3d, line_distance = find_closest_point_between_lines(
            laser_pos, laser_dir,
            camera_pos, cam_dir
        )

        results.append([
            idx_x, idx_y,
            point_3d[0], point_3d[1], point_3d[2],
            u, v,
            line_distance
        ])

    if len(results) == 0:
        return np.empty((0, 8), dtype=np.float32)

    return np.array(results, dtype=np.float32)


# ============================================================
# TRAJECTORY TRIANGULATION
# ============================================================
def triangulate_single_trajectory_point(
    u: float,
    v: float,
    laser_pos: np.ndarray,
    metadata: dict
) -> tuple[np.ndarray, float]:
    """
    Trianguliert einen einzelnen Trajectory-Messpunkt.

    Der Laserstrahl wird durch:
    - die Laserposition des aktuellen Frames
    - die globale Basisrichtung aus den Metadaten

    beschrieben.

    Returns
    -------
    point_3d : np.ndarray, shape (3,)
    line_distance : float
    """
    camera_pos = np.array([0.0, 0.0, 0.0], dtype=float)

    laser_pos = np.asarray(laser_pos, dtype=float)
    laser_dir = get_laser_base_direction_from_metadata(metadata)
    cam_dir = get_camera_ray_from_pixel(u, v, metadata)

    point_3d, line_distance = find_closest_point_between_lines(
        laser_pos, laser_dir,
        camera_pos, cam_dir
    )

    return point_3d, line_distance


def triangulate_trajectory_uv_points(
    uv_points: np.ndarray,
    frame_rows_by_idx: dict[int, dict],
    metadata: dict
) -> np.ndarray:
    """
    Trianguliert trajectory-basierte Punkte.

    Erwartetes Format von uv_points:
        [u, v, frame_idx]

    Rückgabeformat:
        [x, y, z, u, v, frame_idx]
    """
    results = []

    for row in uv_points:
        u = float(row[0])
        v = float(row[1])
        frame_idx = int(row[2])

        if frame_idx not in frame_rows_by_idx:
            raise KeyError(f"frame_idx {frame_idx} nicht in frame_table gefunden.")

        frame_row = frame_rows_by_idx[frame_idx]
        laser_pos = np.array([
            frame_row["laser_x"],
            frame_row["laser_y"],
            frame_row["laser_z"],
        ], dtype=float)

        point_3d, _line_distance = triangulate_single_trajectory_point(
            u=u,
            v=v,
            laser_pos=laser_pos,
            metadata=metadata
        )

        results.append([
            point_3d[0],
            point_3d[1],
            point_3d[2],
            u,
            v,
            frame_idx
        ])

    if len(results) == 0:
        return np.empty((0, 6), dtype=np.float32)

    return np.array(results, dtype=np.float32)