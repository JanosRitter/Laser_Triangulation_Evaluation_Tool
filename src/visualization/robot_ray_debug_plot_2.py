from __future__ import annotations

from pathlib import Path
import csv

import numpy as np
import matplotlib.pyplot as plt

from src.triangulation.laser_rays.robot_trajectory_rays import (
    LOCAL_LASER_DIRECTION_R,
    LOCAL_LASER_ORIGIN_OFFSET_L,
    _frame_row_to_robot_pose,
    _robot_pose_to_laser_ray_R,
)


# ============================================================
# USER SETTINGS
# ============================================================

RUN_DIR = Path(
    r"C:\Users\JRI\Documents\Robotercode_new\laser_triangulation\data\input\images\20260513_140338_robot_measurement"
)

FRAME_TABLE_PATH = RUN_DIR / "frame_table.csv"

RAY_LENGTH_M = 0.15
MAX_RAYS_TO_PLOT = 100

Z_PLANE_M = 0.50

SAVE_PATH_3D = RUN_DIR / "debug_laser_rays_robot_frame_R.png"
SAVE_PATH_Z_PLANE_2D = RUN_DIR / "debug_laser_rays_z_plane_xy_R.png"

SHOW_PLOTS = True


# ============================================================
# Helpers
# ============================================================

def load_frame_table(path: Path) -> list[dict]:
    with open(path, "r", encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def select_valid_rows(rows: list[dict]) -> list[dict]:
    valid_rows = [r for r in rows if r.get("status") == "valid"]

    if len(valid_rows) > MAX_RAYS_TO_PLOT:
        idx = np.linspace(0, len(valid_rows) - 1, MAX_RAYS_TO_PLOT).astype(int)
        return [valid_rows[i] for i in idx]

    return valid_rows


def frame_row_to_laser_ray_R(row: dict):
    laser_position_R, laser_rotation_deg, frame_idx = _frame_row_to_robot_pose(row)

    origin_R, direction_R = _robot_pose_to_laser_ray_R(
        laser_position_R=laser_position_R,
        laser_rotation_deg=laser_rotation_deg,
        local_laser_direction=LOCAL_LASER_DIRECTION_R,
        local_laser_origin_offset=LOCAL_LASER_ORIGIN_OFFSET_L,
    )

    end_R = origin_R + RAY_LENGTH_M * direction_R

    return frame_idx, laser_position_R, origin_R, direction_R, end_R


def intersect_ray_with_z_plane(
    origin: np.ndarray,
    direction: np.ndarray,
    z_plane: float,
) -> np.ndarray | None:
    origin = np.asarray(origin, dtype=float).reshape(3)
    direction = np.asarray(direction, dtype=float).reshape(3)

    if abs(direction[2]) <= 1e-12:
        return None

    scale = (z_plane - origin[2]) / direction[2]
    return origin + scale * direction


def build_laser_ray_debug_data(rows_to_plot: list[dict]) -> list[dict]:
    data = []

    for row in rows_to_plot:
        frame_idx, position_R, origin_R, direction_R, end_R = frame_row_to_laser_ray_R(row)

        z_plane_point_R = intersect_ray_with_z_plane(
            origin=origin_R,
            direction=direction_R,
            z_plane=Z_PLANE_M,
        )

        data.append(
            {
                "frame_idx": frame_idx,
                "position_R": position_R,
                "origin_R": origin_R,
                "direction_R": direction_R,
                "end_R": end_R,
                "z_plane_point_R": z_plane_point_R,
            }
        )

    return data


def set_axes_equal(ax, points: np.ndarray) -> None:
    x = points[:, 0]
    y = points[:, 1]
    z = points[:, 2]

    cx = 0.5 * (np.min(x) + np.max(x))
    cy = 0.5 * (np.min(y) + np.max(y))
    cz = 0.5 * (np.min(z) + np.max(z))

    span = max(
        np.max(x) - np.min(x),
        np.max(y) - np.min(y),
        np.max(z) - np.min(z),
        1e-6,
    )

    half = 0.5 * span

    ax.set_xlim(cx - half, cx + half)
    ax.set_ylim(cy - half, cy + half)
    ax.set_zlim(cz - half, cz + half)


# ============================================================
# Plot 1: 3D rays wie bisher
# ============================================================

def plot_laser_rays_3d(debug_data: list[dict]) -> None:
    fig = plt.figure(figsize=(9, 7))
    ax = fig.add_subplot(111, projection="3d")
    ax.set_proj_type("ortho")

    all_points = []

    for item in debug_data:
        position_R = item["position_R"]
        origin_R = item["origin_R"]
        end_R = item["end_R"]

        ax.scatter(
            position_R[0],
            position_R[1],
            position_R[2],
            s=12,
            marker="o",
            alpha=0.55,
        )

        ax.scatter(
            origin_R[0],
            origin_R[1],
            origin_R[2],
            s=18,
            marker="^",
            alpha=0.8,
        )

        ax.plot(
            [position_R[0], origin_R[0]],
            [position_R[1], origin_R[1]],
            [position_R[2], origin_R[2]],
            linewidth=0.7,
            alpha=0.6,
        )

        ax.plot(
            [origin_R[0], end_R[0]],
            [origin_R[1], end_R[1]],
            [origin_R[2], end_R[2]],
            linewidth=1.0,
            alpha=0.75,
        )

        all_points.extend([position_R, origin_R, end_R])

    all_points = np.asarray(all_points, dtype=float)
    set_axes_equal(ax, all_points)

    ax.set_title("Laser ray construction in robot frame R")
    ax.set_xlabel("x_R [m]")
    ax.set_ylabel("y_R [m]")
    ax.set_zlabel("z_R [m]")

    ax.scatter([], [], [], marker="o", label="Robot pose / flange point")
    ax.scatter([], [], [], marker="^", label="Laser origin after offset")
    ax.plot([], [], [], label="Laser ray")

    ax.legend()
    ax.grid(True, linestyle=":", linewidth=0.5, alpha=0.5)

    plt.tight_layout()

    SAVE_PATH_3D.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(SAVE_PATH_3D, dpi=300)
    print(f"\n🖼️ 3D-Debugplot gespeichert: {SAVE_PATH_3D}")

    if SHOW_PLOTS:
        plt.show()
    else:
        plt.close(fig)


# ============================================================
# Plot 2: 2D-Schnittpunkte mit z-Ebene
# ============================================================

def plot_laser_ray_z_plane_intersections_2d(debug_data: list[dict]) -> None:
    points = []
    frame_indices = []

    for item in debug_data:
        p = item["z_plane_point_R"]

        if p is None:
            continue

        points.append(p)
        frame_indices.append(item["frame_idx"])

    if len(points) == 0:
        print(f"\n⚠️ Keine Schnittpunkte mit z={Z_PLANE_M:.3f} gefunden.")
        return

    points = np.asarray(points, dtype=float)

    print("\n📍 Laser-Schnittpunkte mit z-Ebene:")
    print(f"  z = {Z_PLANE_M:.6f} m")
    print(f"  Anzahl: {len(points)}")
    print(f"  x min/max: {np.min(points[:, 0]):+.6f} / {np.max(points[:, 0]):+.6f}")
    print(f"  y min/max: {np.min(points[:, 1]):+.6f} / {np.max(points[:, 1]):+.6f}")

    fig, ax = plt.subplots(figsize=(8, 8))

    ax.scatter(
        points[:, 0],
        points[:, 1],
        s=28,
        marker="x",
        label=f"Laser ray @ z={Z_PLANE_M:.2f} m",
    )

    for frame_idx, p in zip(frame_indices, points):
        ax.text(
            p[0],
            p[1],
            str(frame_idx),
            fontsize=7,
            alpha=0.75,
        )

    ax.set_title(f"Laser ray intersections with z={Z_PLANE_M:.2f} m in robot frame R")
    ax.set_xlabel("x_R [m]")
    ax.set_ylabel("y_R [m]")
    ax.axis("equal")
    ax.grid(True, linestyle=":", linewidth=0.5, alpha=0.6)
    ax.legend()

    plt.tight_layout()

    SAVE_PATH_Z_PLANE_2D.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(SAVE_PATH_Z_PLANE_2D, dpi=300)
    print(f"🖼️ 2D-Schnittpunktplot gespeichert: {SAVE_PATH_Z_PLANE_2D}")

    if SHOW_PLOTS:
        plt.show()
    else:
        plt.close(fig)


# ============================================================
# Main
# ============================================================

def main() -> None:
    rows = load_frame_table(FRAME_TABLE_PATH)
    rows_to_plot = select_valid_rows(rows)

    print("\n🔦 Laser-Ray-Robot-Frame-Debug")
    print(f"  frame_table: {FRAME_TABLE_PATH}")
    print(f"  rows total:  {len(rows)}")
    print(f"  plotted:     {len(rows_to_plot)}")
    print(f"  local direction: {LOCAL_LASER_DIRECTION_R}")
    print(f"  local offset:    {LOCAL_LASER_ORIGIN_OFFSET_L}")
    print(f"  z-plane:         {Z_PLANE_M:.6f} m")

    debug_data = build_laser_ray_debug_data(rows_to_plot)

    if debug_data:
        first = debug_data[0]
        print("\nBeispiel erster geplotter Frame:")
        print(f"  frame_idx:   {first['frame_idx']}")
        print(f"  position_R:  {first['position_R']}")
        print(f"  origin_R:    {first['origin_R']}")
        print(f"  direction_R: {first['direction_R']}")
        print(f"  end_R:       {first['end_R']}")
        print(f"  z-plane p:   {first['z_plane_point_R']}")

    plot_laser_rays_3d(debug_data)
    plot_laser_ray_z_plane_intersections_2d(debug_data)


if __name__ == "__main__":
    main()