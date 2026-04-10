from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.tri as mtri
import numpy as np

from .plot_utils import _set_axes_3d_compact


def plot_surface_from_triangulated_points(
    triangulated_points: np.ndarray,
    save_path: str | Path | None = None,
    show: bool = True,
    elev: float = 25,
    azim: float = -60,
    invert_z: bool = True,
    cmap: str = "viridis",
    xyz_cols: tuple[int, int, int] = (2, 3, 4),
    z_window: tuple[float, float] | None = None,
    box_aspect: tuple[float, float, float] = (1.0, 1.0, 0.5),
    orthographic: bool = True,
    title: str = "Surface Plot",
):
    """
    Minimalversion:
    - erwartet ein Array der Form z.B.
      [frame_idx, unused, x, y, z, u, v]
      oder
      [x_ind, y_ind, x, y, z, u, v]
    - verwendet nur x, y, z
    - trianguliert automatisch in der XY-Ebene
    - plottet die Oberfläche mit plot_trisurf
    """
    triangulated_points = np.asarray(triangulated_points)

    if triangulated_points.ndim != 2:
        raise ValueError("triangulated_points muss ein 2D-Array sein.")

    req_cols = max(*xyz_cols) + 1
    if triangulated_points.shape[1] < req_cols:
        raise ValueError(
            f"triangulated_points muss mindestens {req_cols} Spalten besitzen."
        )

    xyz = triangulated_points[:, xyz_cols].astype(float)

    finite_mask = np.all(np.isfinite(xyz), axis=1)
    xyz = xyz[finite_mask]

    if len(xyz) < 3:
        raise ValueError("Zu wenige gültige Punkte für eine Oberfläche.")

    x = xyz[:, 0]
    y = xyz[:, 1]
    z = xyz[:, 2]

    fig = plt.figure(figsize=(8.0, 5.8))
    ax = fig.add_subplot(111, projection="3d")

    if orthographic:
        ax.set_proj_type("ortho")

    tri = mtri.Triangulation(x, y)

    surf = ax.plot_trisurf(
        x,
        y,
        z,
        triangles=tri.triangles,
        cmap=cmap,
        linewidth=0.0,
        edgecolor="none",
        antialiased=True,
        shade=False,
        alpha=1.0,
    )

    pos = ax.get_position()
    cbar_width = 0.018
    cbar_pad = 0.13
    cbar_height = pos.height * 0.75
    cbar_y = pos.y0 + (pos.height - cbar_height) / 2

    cax = fig.add_axes([
        pos.x1 + cbar_pad,
        cbar_y,
        cbar_width,
        cbar_height,
    ])
    cbar = fig.colorbar(surf, cax=cax)
    cbar.set_label("z [m]")

    _set_axes_3d_compact(
        ax,
        x,
        y,
        z,
        xy_margin_factor=0.06,
        z_margin_factor=0.03,
        z_window=z_window,
        box_aspect=box_aspect,
    )

    if invert_z:
        ax.invert_zaxis()

    ax.set_xlabel("x [m]", labelpad=8)
    ax.set_ylabel("y [m]", labelpad=8)
    ax.set_zlabel("z [m]", labelpad=1)
    ax.set_title(title)

    ax.view_init(elev=elev, azim=azim)
    ax.grid(True, linestyle=":", linewidth=0.5, alpha=0.35)
    ax.xaxis.pane.fill = False
    ax.yaxis.pane.fill = False
    ax.zaxis.pane.fill = False

    plt.tight_layout()

    if save_path is not None:
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=350, bbox_inches="tight")
        print(f"  🖼️ Surface-Plot gespeichert: {save_path}")

    if show:
        plt.show()
    else:
        plt.close(fig)

    return {
        "xyz": xyz,
        "triangles": tri.triangles,
        "n_input_points": int(len(triangulated_points)),
        "n_valid_points": int(len(xyz)),
        "n_triangles": int(len(tri.triangles)),
    }