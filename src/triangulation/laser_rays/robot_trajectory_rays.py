from __future__ import annotations

from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation

from src.triangulation.metadata_io import load_robot_to_camera_calibration


_HAS_PRINTED_CALIBRATION_INFO = False


REQUIRED_FRAME_COLUMNS = [
    "frame_idx",
    "laser_x",
    "laser_y",
    "laser_z",
    "laser_rx",
    "laser_ry",
    "laser_rz",
]


# Muss zur Kalibrierung passen:
# Im Kalibrierungstool war local_direction = np.array([0.0, 1.0, 0.0])
LOCAL_LASER_DIRECTION_R = np.array([1.0, 0.0, 0.0], dtype=float)
LOCAL_LASER_ORIGIN_OFFSET_L = np.array([0.0, 0.0, 0.042], dtype=float)


def _normalize(v: np.ndarray) -> np.ndarray:
    v = np.asarray(v, dtype=float).reshape(3)
    norm = np.linalg.norm(v)

    if norm <= 1e-15:
        raise ValueError("Vektor darf nicht null sein.")

    return v / norm


def _check_frame_row(frame_row: dict) -> None:
    missing = [
        key for key in REQUIRED_FRAME_COLUMNS
        if key not in frame_row
    ]

    if missing:
        raise KeyError(
            "frame_row enthält nicht alle benötigten Spalten. "
            f"Fehlend: {missing}"
        )

    for key in REQUIRED_FRAME_COLUMNS:
        value = frame_row[key]
        try:
            float(value)
        except Exception as exc:
            raise ValueError(
                f"frame_row[{key!r}] ist nicht numerisch: {value!r}"
            ) from exc


def _print_calibration_summary(calibration: dict) -> None:
    global _HAS_PRINTED_CALIBRATION_INFO

    if _HAS_PRINTED_CALIBRATION_INFO:
        return

    _HAS_PRINTED_CALIBRATION_INFO = True

    calibration_path = calibration.get("_calibration_path", "<unknown>")
    T_C_R = calibration["_T_C_R_array"]

    print("\n📐 Robot-to-Camera-Kalibrierung geladen")
    print(f"  Datei: {calibration_path}")
    print(f"  T_C_R shape: {T_C_R.shape}")

    print("\n  Transformationskonvention:")
    convention = calibration.get("transformation_convention", {})
    print(f"    Punkt:     {convention.get('point_transform', '<fehlt>')}")
    print(f"    Richtung:  {convention.get('direction_transform', '<fehlt>')}")

    print("\n  Kamera-KS:")
    camera_frame = calibration.get("camera_frame_convention", {})
    print(f"    x_C: {camera_frame.get('x_C', '<fehlt>')}")
    print(f"    y_C: {camera_frame.get('y_C', '<fehlt>')}")
    print(f"    z_C: {camera_frame.get('z_C', '<fehlt>')}")
    print(f"    viewing_direction: {camera_frame.get('viewing_direction', '<fehlt>')}")

    metadata = calibration.get("metadata", {})
    intrinsics = metadata.get("intrinsics", {})

    if intrinsics:
        print("\n  Intrinsics aus Kalibrierdatei:")
        print(f"    source: {intrinsics.get('source', '<fehlt>')}")
        print(f"    fx: {intrinsics.get('fx', '<fehlt>')}")
        print(f"    fy: {intrinsics.get('fy', '<fehlt>')}")
        print(f"    cx: {intrinsics.get('cx', '<fehlt>')}")
        print(f"    cy: {intrinsics.get('cy', '<fehlt>')}")
        print(f"    img_width: {intrinsics.get('img_width', '<fehlt>')}")
        print(f"    img_height: {intrinsics.get('img_height', '<fehlt>')}")
        print(f"    dist_coeffs: {intrinsics.get('dist_coeffs', '<fehlt>')}")
    else:
        print("\n  ⚠️ Keine Intrinsics-Metadaten in der Kalibrierdatei gefunden.")


def _frame_row_to_robot_pose(frame_row: dict) -> tuple[np.ndarray, np.ndarray, int]:
    """
    Liest die reale Roboterpose aus frame_table.csv.

    Rückgabe:
        laser_position_R
        laser_rotation_deg
        frame_idx
    """
    _check_frame_row(frame_row)

    frame_idx = int(float(frame_row["frame_idx"]))

    laser_position_R = np.array(
        [
            float(frame_row["laser_x"]),
            float(frame_row["laser_y"]),
            float(frame_row["laser_z"]),
        ],
        dtype=float,
    )

    laser_rotation_deg = np.array(
        [
            float(frame_row["laser_rx"]),
            float(frame_row["laser_ry"]),
            float(frame_row["laser_rz"]),
        ],
        dtype=float,
    )

    return laser_position_R, laser_rotation_deg, frame_idx


def _robot_pose_to_laser_ray_R(
    laser_position_R: np.ndarray,
    laser_rotation_deg: np.ndarray,
    local_laser_direction: np.ndarray = LOCAL_LASER_DIRECTION_R,
    local_laser_origin_offset: np.ndarray = LOCAL_LASER_ORIGIN_OFFSET_L,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Baut den Laserstrahl im Roboter-KS R.

    laser_position_R:
        gespeicherte Roboter-/Flansch-/Montageplattenposition im Roboter-KS

    local_laser_origin_offset:
        Offset vom gespeicherten Roboterpunkt zum echten Laserursprung,
        ausgedrückt im lokalen Tool-/Platten-KS.

    direction_R = R_R_L @ direction_L
    origin_R    = position_R + R_R_L @ offset_L
    """
    laser_position_R = np.asarray(laser_position_R, dtype=float).reshape(3)
    laser_rotation_deg = np.asarray(laser_rotation_deg, dtype=float).reshape(3)

    R_R_L = Rotation.from_euler(
        "xyz",
        laser_rotation_deg,
        degrees=True,
    ).as_matrix()

    offset_L = np.asarray(local_laser_origin_offset, dtype=float).reshape(3)

    laser_origin_R = laser_position_R + R_R_L @ offset_L

    direction_L = _normalize(local_laser_direction)
    direction_R = R_R_L @ direction_L
    direction_R = _normalize(direction_R)

    return laser_origin_R, direction_R


def _transform_laser_ray_R_to_C(
    origin_R: np.ndarray,
    direction_R: np.ndarray,
    calibration: dict,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Transformiert einen Laserstrahl aus dem Roboter-KS R ins Kamera-KS C.

    Kalibrierdatei-Konvention:
        p_C = R_C_R @ p_R + t_C_R
        d_C = R_C_R @ d_R
    """
    T_C_R = np.asarray(calibration["_T_C_R_array"], dtype=float).reshape(4, 4)

    R_C_R = T_C_R[:3, :3]
    t_C_R = T_C_R[:3, 3]

    origin_R = np.asarray(origin_R, dtype=float).reshape(3)
    direction_R = _normalize(direction_R)

    origin_C = R_C_R @ origin_R + t_C_R
    direction_C = R_C_R @ direction_R
    direction_C = _normalize(direction_C)

    return origin_C, direction_C


def get_robot_trajectory_laser_ray_from_frame_row(
    frame_row: dict,
    calibration_path: str | Path | None = None,
    calibration: dict | None = None,
    local_laser_direction: np.ndarray = LOCAL_LASER_DIRECTION_R,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Rekonstruiert den Laserstrahl eines Frames im Kamera-KS.

    Input:
        frame_row:
            Zeile aus frame_table.csv mit:
            laser_x, laser_y, laser_z, laser_rx, laser_ry, laser_rz

        calibration:
            geladene robot_to_camera_transform.json

    Output:
        laser_origin_C:
            Ursprung des Laserstrahls im Kamera-KS

        laser_direction_C:
            Richtung des Laserstrahls im Kamera-KS
    """
    if calibration is None:
        calibration = load_robot_to_camera_calibration(calibration_path)

    _print_calibration_summary(calibration)

    laser_position_R, laser_rotation_deg, frame_idx = _frame_row_to_robot_pose(frame_row)

    laser_origin_R, laser_direction_R = _robot_pose_to_laser_ray_R(
        laser_position_R=laser_position_R,
        laser_rotation_deg=laser_rotation_deg,
        local_laser_direction=local_laser_direction,
        local_laser_origin_offset=LOCAL_LASER_ORIGIN_OFFSET_L,
    )

    laser_origin_C, laser_direction_C = _transform_laser_ray_R_to_C(
        origin_R=laser_origin_R,
        direction_R=laser_direction_R,
        calibration=calibration,
    )

    print(f"\n🔦 Frame {frame_idx:06d}: Laser-Ray rekonstruiert")
    print(
        f"  origin_R = "
        f"({laser_origin_R[0]:+.6f}, "
        f"{laser_origin_R[1]:+.6f}, "
        f"{laser_origin_R[2]:+.6f}) m"
    )
    print(
        f"  dir_R    = "
        f"({laser_direction_R[0]:+.6f}, "
        f"{laser_direction_R[1]:+.6f}, "
        f"{laser_direction_R[2]:+.6f})"
    )
    print(
        f"  origin_C = "
        f"({laser_origin_C[0]:+.6f}, "
        f"{laser_origin_C[1]:+.6f}, "
        f"{laser_origin_C[2]:+.6f}) m"
    )
    print(
        f"  dir_C    = "
        f"({laser_direction_C[0]:+.6f}, "
        f"{laser_direction_C[1]:+.6f}, "
        f"{laser_direction_C[2]:+.6f})"
    )

    return laser_origin_C, laser_direction_C