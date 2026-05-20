from __future__ import annotations

from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt


# ============================================================
# USER SETTINGS
# ============================================================

OUTPUT_RUN_DIR = Path(
    r"C:\Users\JRI\Documents\Robotercode_new\laser_triangulation\data\output\20260513_140338_robot_measurement"
)

CAMERA_RAYS_FILE = "robot_trajectory_camera_rays_C.npy"
LASER_RAYS_FILE = "robot_trajectory_laser_rays_C.npy"

RAY_LENGTH_M = 0.4
MAX_RAYS_TO_PLOT = 100
Z_PLANE_M = -0.362
SAVE_Z_PLANE_PATH = OUTPUT_RUN_DIR / "robot_trajectory_ray_intersections_z_-0p4_C.png"

SAVE_PATH = OUTPUT_RUN_DIR / "robot_trajectory_ray_debug_C.png"
SHOW_PLOT = True


# ============================================================
# Helpers
# ============================================================

def load_rays(path: Path) -> np.ndarray:
    """
    Erwartetes Format:
    [frame_idx,
     origin_x, origin_y, origin_z,
     direction_x, direction_y, direction_z,
     u, v]
    """
    rays = np.load(path)

    if rays.ndim != 2 or rays.shape[1] != 9:
        raise ValueError(
            f"Rays müssen Shape (n,9) haben, haben aber {rays.shape}: {path}"
        )

    return rays.astype(float)


def closest_points_between_lines(
    p1: np.ndarray,
    d1: np.ndarray,
    p2: np.ndarray,
    d2: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, float]:
    d1 = d1 / np.linalg.norm(d1)
    d2 = d2 / np.linalg.norm(d2)

    diff = p1 - p2

    a = np.dot(d1, d1)
    b = np.dot(d1, d2)
    c = np.dot(d2, d2)
    d = np.dot(d1, diff)
    e = np.dot(d2, diff)

    denom = a * c - b * b

    if np.isclose(denom, 0.0):
        return p1, p2, np.nan

    s = (b * e - c * d) / denom
    t = (a * e - b * d) / denom

    q1 = p1 + s * d1
    q2 = p2 + t * d2

    return q1, q2, float(np.linalg.norm(q1 - q2))


def set_axes_equal(ax, points: np.ndarray) -> None:
    x = points[:, 0]
    y = points[:, 1]
    z = points[:, 2]

    x_mid = 0.5 * (np.min(x) + np.max(x))
    y_mid = 0.5 * (np.min(y) + np.max(y))
    z_mid = 0.5 * (np.min(z) + np.max(z))

    span = max(
        np.max(x) - np.min(x),
        np.max(y) - np.min(y),
        np.max(z) - np.min(z),
        1e-6,
    )

    half = 0.5 * span

    ax.set_xlim(x_mid - half, x_mid + half)
    ax.set_ylim(y_mid - half, y_mid + half)
    ax.set_zlim(z_mid - half, z_mid + half)
    
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

    # Für Debugging erstmal NICHT verwerfen.
    # Wenn scale < 0 ist, liegt der Schnittpunkt "hinter" der Ray-Richtung.
    point = origin + scale * direction
    return point


def plot_saved_rays() -> None:
    camera_rays = load_rays(OUTPUT_RUN_DIR / CAMERA_RAYS_FILE)
    laser_rays = load_rays(OUTPUT_RUN_DIR / LASER_RAYS_FILE)

    camera_by_frame = {
        int(row[0]): row
        for row in camera_rays
    }

    laser_by_frame = {
        int(row[0]): row
        for row in laser_rays
    }

    common_frames = sorted(set(camera_by_frame) & set(laser_by_frame))

    if len(common_frames) == 0:
        raise RuntimeError("Keine gemeinsamen frame_idx zwischen Kamera- und Laserrays gefunden.")

    if len(common_frames) > MAX_RAYS_TO_PLOT:
        idx = np.linspace(0, len(common_frames) - 1, MAX_RAYS_TO_PLOT).astype(int)
        frames_to_plot = [common_frames[i] for i in idx]
    else:
        frames_to_plot = common_frames

    print("\n📊 Ray-Debug")
    print(f"  Camera rays: {len(camera_rays)}")
    print(f"  Laser rays:  {len(laser_rays)}")
    print(f"  Common:      {len(common_frames)}")
    print(f"  Plotted:     {len(frames_to_plot)}")

    fig = plt.figure(figsize=(9, 7))
    ax = fig.add_subplot(111, projection="3d")
    ax.set_proj_type("ortho")

    all_points = []
    distances = []

    for frame_idx in frames_to_plot:
        c = camera_by_frame[frame_idx]
        l = laser_by_frame[frame_idx]

        c_origin = c[1:4]
        c_dir = c[4:7]
        c_dir = c_dir / np.linalg.norm(c_dir)

        l_origin = l[1:4]
        l_dir = l[4:7]
        l_dir = l_dir / np.linalg.norm(l_dir)
        
        # Laser-Startpunkt / Ursprung
        ax.scatter(
            l_origin[0],
            l_origin[1],
            l_origin[2],
            s=18,
            marker="^",
            alpha=0.85,
        )

        c_end = c_origin + RAY_LENGTH_M * c_dir
        l_end = l_origin + RAY_LENGTH_M * l_dir

        q_c, q_l, dist = closest_points_between_lines(
            p1=c_origin,
            d1=c_dir,
            p2=l_origin,
            d2=l_dir,
        )

        distances.append(dist)

        # Kamera-Ray
        ax.plot(
            [c_origin[0], c_end[0]],
            [c_origin[1], c_end[1]],
            [c_origin[2], c_end[2]],
            linewidth=0.8,
            alpha=0.55,
        )

        # Laser-Ray
        ax.plot(
            [l_origin[0], l_end[0]],
            [l_origin[1], l_end[1]],
            [l_origin[2], l_end[2]],
            linewidth=0.8,
            alpha=0.55,
        )

        # Kürzeste Verbindung
        if np.isfinite(dist):
            ax.plot(
                [q_c[0], q_l[0]],
                [q_c[1], q_l[1]],
                [q_c[2], q_l[2]],
                linewidth=0.6,
                alpha=0.4,
            )

        all_points.extend([c_origin, c_end, l_origin, l_end, q_c, q_l])

    all_points = np.asarray(all_points, dtype=float)
    distances = np.asarray(distances, dtype=float)

    print("\n📏 Ray-Abstände:")
    print(f"  mean:   {np.nanmean(distances) * 1000.0:.3f} mm")
    print(f"  median: {np.nanmedian(distances) * 1000.0:.3f} mm")
    print(f"  max:    {np.nanmax(distances) * 1000.0:.3f} mm")

    ax.scatter([0], [0], [0], s=50, marker="o", label="Camera origin C")

    ax.set_title("Camera rays and laser rays in camera frame C")
    ax.set_xlabel("x_C [m]")
    ax.set_ylabel("y_C [m]")
    ax.set_zlabel("z_C [m]")

    set_axes_equal(ax, all_points)

    ax.legend()
    ax.grid(True, linestyle=":", linewidth=0.5, alpha=0.5)

    plt.tight_layout()

    SAVE_PATH.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(SAVE_PATH, dpi=300)
    print(f"\n🖼️ Ray-Debug gespeichert: {SAVE_PATH}")

    if SHOW_PLOT:
        plt.show()
    else:
        plt.close(fig)
        
    # ============================================================
    # Zusatzplot: Schnittpunkte mit z = Z_PLANE_M
    # ============================================================

    camera_intersections = []
    laser_intersections = []
    pair_distances_xy = []
    plotted_frame_indices = []

    for frame_idx in frames_to_plot:
        c = camera_by_frame[frame_idx]
        l = laser_by_frame[frame_idx]

        c_origin = c[1:4]
        c_dir = c[4:7]
        c_dir = c_dir / np.linalg.norm(c_dir)

        l_origin = l[1:4]
        l_dir = l[4:7]
        l_dir = l_dir / np.linalg.norm(l_dir)

        p_cam = intersect_ray_with_z_plane(
            origin=c_origin,
            direction=c_dir,
            z_plane=Z_PLANE_M,
        )

        p_laser = intersect_ray_with_z_plane(
            origin=l_origin,
            direction=l_dir,
            z_plane=Z_PLANE_M,
        )

        if p_cam is None or p_laser is None:
            continue

        camera_intersections.append(p_cam)
        laser_intersections.append(p_laser)
        pair_distances_xy.append(np.linalg.norm(p_cam[:2] - p_laser[:2]))
        plotted_frame_indices.append(frame_idx)

    camera_intersections = np.asarray(camera_intersections, dtype=float)
    laser_intersections = np.asarray(laser_intersections, dtype=float)
    pair_distances_xy = np.asarray(pair_distances_xy, dtype=float)

    if len(camera_intersections) == 0:
        print(f"\n⚠️ Keine Schnittpunkte mit z={Z_PLANE_M} gefunden.")
        return

    print(f"\n📍 Schnittpunkte mit z = {Z_PLANE_M:.3f} m:")
    print(f"  Paare:      {len(camera_intersections)}")
    print(f"  mean XY:    {np.mean(pair_distances_xy) * 1000.0:.3f} mm")
    print(f"  median XY:  {np.median(pair_distances_xy) * 1000.0:.3f} mm")
    print(f"  max XY:     {np.max(pair_distances_xy) * 1000.0:.3f} mm")

    fig2, ax2 = plt.subplots(figsize=(9, 8))

    # Verbindungslinien gepaarter Schnittpunkte
    for i, frame_idx in enumerate(plotted_frame_indices):
        pc = camera_intersections[i]
        pl = laser_intersections[i]

        ax2.plot(
            [pc[0], pl[0]],
            [pc[1], pl[1]],
            linewidth=0.7,
            alpha=0.55,
        )

        ax2.text(
            0.5 * (pc[0] + pl[0]),
            0.5 * (pc[1] + pl[1]),
            str(frame_idx),
            fontsize=7,
            alpha=0.8,
        )

    scatter = ax2.scatter(
        0.5 * (camera_intersections[:, 0] + laser_intersections[:, 0]),
        0.5 * (camera_intersections[:, 1] + laser_intersections[:, 1]),
        c=pair_distances_xy * 1000.0,
        s=35,
        label="midpoint on z-plane",
    )

    ax2.scatter(
        camera_intersections[:, 0],
        camera_intersections[:, 1],
        s=20,
        marker="+",
        label="camera ray @ z-plane",
    )

    ax2.scatter(
        laser_intersections[:, 0],
        laser_intersections[:, 1],
        s=20,
        marker="x",
        label="laser ray @ z-plane",
    )

    cbar = fig2.colorbar(scatter, ax=ax2)
    cbar.set_label("XY distance camera/laser [mm]")

    ax2.set_title(f"Ray intersections with z = {Z_PLANE_M:.3f} m in camera frame C")
    ax2.set_xlabel("x_C [m]")
    ax2.set_ylabel("y_C [m]")
    ax2.axis("equal")
    ax2.grid(True, linestyle=":", linewidth=0.5, alpha=0.5)
    ax2.legend(loc="best")

    all_xy = np.vstack([
        camera_intersections[:, :2],
        laser_intersections[:, :2],
    ])

    xy_min = np.min(all_xy, axis=0)
    xy_max = np.max(all_xy, axis=0)
    xy_center = 0.5 * (xy_min + xy_max)
    xy_span = np.max(xy_max - xy_min)
    xy_span = max(xy_span, 0.05)

    ax2.set_xlim(xy_center[0] - 0.55 * xy_span, xy_center[0] + 0.55 * xy_span)
    ax2.set_ylim(xy_center[1] - 0.55 * xy_span, xy_center[1] + 0.55 * xy_span)

    fig2.tight_layout()
    SAVE_Z_PLANE_PATH.parent.mkdir(parents=True, exist_ok=True)
    fig2.savefig(SAVE_Z_PLANE_PATH, dpi=300)

    print(f"🖼️ z-Ebenen-Schnittplot gespeichert: {SAVE_Z_PLANE_PATH}")

    if SHOW_PLOT:
        plt.show()
    else:
        plt.close(fig2)
        
    


if __name__ == "__main__":
    plot_saved_rays()