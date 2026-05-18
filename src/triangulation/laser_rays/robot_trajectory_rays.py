from __future__ import annotations

import numpy as np


def get_robot_trajectory_laser_ray_from_frame_row(
    frame_row: dict,
    T_C_R: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Neue echte Roboter-Trajectory-Logik.

    Noch nicht aktiv verwendet.

    Später:
    - vollständige Roboterpose aus frame_table.csv lesen
    - Laser-Ray im Roboter-KS bauen
    - optional mit T_C_R ins Kamera-KS transformieren
    """
    raise NotImplementedError(
        "robot_trajectory_rays.py ist vorbereitet, aber die echte "
        "Roboter-Trajectory-Ray-Logik wird im nächsten Schritt ergänzt."
    )