from __future__ import annotations

import numpy as np

from src.triangulation.laser_rays.doe_rays import (
    get_laser_base_direction_from_metadata,
)


def get_sim_trajectory_laser_ray_from_frame_row(
    frame_row: dict,
    metadata: dict,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Alte Trajectory-Logik.

    Ursprung:
        laser_x, laser_y, laser_z aus frame_table.csv

    Richtung:
        globale Basisrichtung aus metadata["laser"]["rotation_x_deg"]
        und metadata["laser"]["rotation_y_deg"]

    Wichtig:
        laser_rx, laser_ry, laser_rz aus frame_table.csv werden hier
        noch nicht verwendet.
    """
    laser_origin = np.array(
        [
            frame_row["laser_x"],
            frame_row["laser_y"],
            frame_row["laser_z"],
        ],
        dtype=float,
    )

    laser_direction = get_laser_base_direction_from_metadata(metadata)

    return laser_origin, laser_direction