from __future__ import annotations

import numpy as np

from src.triangulation.camera_rays import get_camera_ray_from_pixel
from src.triangulation.laser_rays.doe_rays import get_doe_laser_ray_from_index
from src.triangulation.laser_rays.sim_trajectory_rays import (
    get_sim_trajectory_laser_ray_from_frame_row,
)


def find_closest_point_between_lines(
    point1: np.ndarray,
    direction1: np.ndarray,
    point2: np.ndarray,
    direction2: np.ndarray,
) -> tuple[np.ndarray, float]:
    """
    Berechnet den Mittelpunkt des kürzesten Verbindungsstücks zwischen zwei Geraden.
    """
    point1 = np.asarray(point1, dtype=float).reshape(3)
    point2 = np.asarray(point2, dtype=float).reshape(3)

    d1 = np.asarray(direction1, dtype=float).reshape(3)
    d2 = np.asarray(direction2, dtype=float).reshape(3)

    d1 = d1 / np.linalg.norm(d1)
    d2 = d2 / np.linalg.norm(d2)

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

    return midpoint, float(distance)


def triangulate_ray_pair(
    laser_origin: np.ndarray,
    laser_direction: np.ndarray,
    camera_origin: np.ndarray,
    camera_direction: np.ndarray,
) -> tuple[np.ndarray, float]:
    """
    Generische Triangulation eines Laser-/Kamera-Ray-Paars.
    """
    return find_closest_point_between_lines(
        point1=laser_origin,
        direction1=laser_direction,
        point2=camera_origin,
        direction2=camera_direction,
    )


def triangulate_indexed_points(
    indexed_points: np.ndarray,
    metadata: dict,
) -> np.ndarray:
    """
    Trianguliert DOE-Punkte aus indizierten Bildpunkten.

    Erwartetes Format:
        [idx_x, idx_y, u, v]

    Rückgabeformat:
        [idx_x, idx_y, x, y, z, u, v, line_distance]
    """
    camera_origin = np.array([0.0, 0.0, 0.0], dtype=float)

    results = []

    for row in indexed_points:
        idx_x = int(row[0])
        idx_y = int(row[1])
        u = float(row[2])
        v = float(row[3])

        laser_origin, laser_direction = get_doe_laser_ray_from_index(
            idx_x=idx_x,
            idx_y=idx_y,
            metadata=metadata,
        )

        camera_direction = get_camera_ray_from_pixel(
            u=u,
            v=v,
            metadata=metadata,
        )

        point_3d, line_distance = triangulate_ray_pair(
            laser_origin=laser_origin,
            laser_direction=laser_direction,
            camera_origin=camera_origin,
            camera_direction=camera_direction,
        )

        results.append(
            [
                idx_x,
                idx_y,
                point_3d[0],
                point_3d[1],
                point_3d[2],
                u,
                v,
                line_distance,
            ]
        )

    if len(results) == 0:
        return np.empty((0, 8), dtype=np.float32)

    return np.array(results, dtype=np.float32)


def triangulate_single_trajectory_point(
    u: float,
    v: float,
    laser_pos: np.ndarray,
    metadata: dict,
) -> tuple[np.ndarray, float]:
    """
    Kompatibilitätsfunktion für die alte Trajectory-Logik.

    Achtung:
        Diese Funktion nutzt wie bisher nur laser_pos und die globale
        Laserbasisrichtung aus metadata.
    """
    camera_origin = np.array([0.0, 0.0, 0.0], dtype=float)

    fake_frame_row = {
        "laser_x": float(laser_pos[0]),
        "laser_y": float(laser_pos[1]),
        "laser_z": float(laser_pos[2]),
    }

    laser_origin, laser_direction = get_sim_trajectory_laser_ray_from_frame_row(
        frame_row=fake_frame_row,
        metadata=metadata,
    )

    camera_direction = get_camera_ray_from_pixel(
        u=u,
        v=v,
        metadata=metadata,
    )

    return triangulate_ray_pair(
        laser_origin=laser_origin,
        laser_direction=laser_direction,
        camera_origin=camera_origin,
        camera_direction=camera_direction,
    )


def triangulate_trajectory_uv_points(
    uv_points: np.ndarray,
    frame_rows_by_idx: dict[int, dict],
    metadata: dict,
) -> np.ndarray:
    """
    Trianguliert alte trajectory-basierte Punkte.

    Erwartetes Format:
        [u, v, frame_idx]

    Rückgabeformat:
        [x, y, z, u, v, frame_idx]
    """
    results = []

    camera_origin = np.array([0.0, 0.0, 0.0], dtype=float)

    for row in uv_points:
        u = float(row[0])
        v = float(row[1])
        frame_idx = int(row[2])

        if frame_idx not in frame_rows_by_idx:
            raise KeyError(f"frame_idx {frame_idx} nicht in frame_table gefunden.")

        frame_row = frame_rows_by_idx[frame_idx]

        laser_origin, laser_direction = get_sim_trajectory_laser_ray_from_frame_row(
            frame_row=frame_row,
            metadata=metadata,
        )

        camera_direction = get_camera_ray_from_pixel(
            u=u,
            v=v,
            metadata=metadata,
        )

        point_3d, _line_distance = triangulate_ray_pair(
            laser_origin=laser_origin,
            laser_direction=laser_direction,
            camera_origin=camera_origin,
            camera_direction=camera_direction,
        )

        results.append(
            [
                point_3d[0],
                point_3d[1],
                point_3d[2],
                u,
                v,
                frame_idx,
            ]
        )

    if len(results) == 0:
        return np.empty((0, 6), dtype=np.float32)

    return np.array(results, dtype=np.float32)