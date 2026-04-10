from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt


def flip_y(coords: np.ndarray, image_height: int) -> np.ndarray:
    """
    Wandelt Bildkoordinaten (Ursprung oben links, y nach unten)
    in mathematische Plotkoordinaten (y nach oben) um.
    """
    flipped = coords.copy()
    flipped[:, 1] = image_height - coords[:, 1]
    return flipped


def plot_image_with_peaks_and_fit(
    image: np.ndarray,
    peaks: np.ndarray,
    fitted_centers: np.ndarray = None,
    title: str = "Detection + Fit",
    save_path: str | Path | None = None,
    show: bool = True,
    zoom: bool = True,
    margin: int = 50
):
    """
    Plot:
    - Bild (Contour)
    - Peaks
    - Fitted Centers
    - optional automatischer Zoom

    Darstellung:
    - x nach rechts
    - y nach oben
    """

    fig, ax = plt.subplots(figsize=(10, 6))

    image_height, image_width = image.shape

    # ---------------------------
    # Contour Plot
    # ---------------------------
    x = np.arange(image_width)
    y = np.arange(image_height)
    y_plot = image_height - y
    xx, yy = np.meshgrid(x, y_plot)

    contour = ax.contourf(xx, yy, image, levels=50, cmap="gray")
    fig.colorbar(contour, ax=ax, label="Intensity")

    # ---------------------------
    # Peaks
    # ---------------------------
    peaks_plot = None
    if peaks is not None and len(peaks) > 0:
        peaks_plot = flip_y(peaks, image_height)

        ax.scatter(
            peaks_plot[:, 0],
            peaks_plot[:, 1],
            marker="x",
            s=40,
            linewidths=1.5,
            label=f"Peaks ({len(peaks)})"
        )

    # ---------------------------
    # Fitted Centers
    # ---------------------------
    fitted_plot = None
    if fitted_centers is not None and len(fitted_centers) > 0:
        fitted_plot = flip_y(fitted_centers, image_height)

        ax.scatter(
            fitted_plot[:, 0],
            fitted_plot[:, 1],
            marker="+",
            s=40,
            linewidths=1.5,
            label=f"Fitted centers ({len(fitted_centers)})"
        )

    # ---------------------------
    # Zoom auf relevanten Bereich
    # ---------------------------
    zoom_source = None
    if fitted_plot is not None and len(fitted_plot) > 0:
        zoom_source = fitted_plot
    elif peaks_plot is not None and len(peaks_plot) > 0:
        zoom_source = peaks_plot

    if zoom and zoom_source is not None and len(zoom_source) > 0:
        x_min = int(np.min(zoom_source[:, 0]) - margin)
        x_max = int(np.max(zoom_source[:, 0]) + margin)
        y_min = int(np.min(zoom_source[:, 1]) - margin)
        y_max = int(np.max(zoom_source[:, 1]) + margin)

        x_min = max(x_min, 0)
        y_min = max(y_min, 0)
        x_max = min(x_max, image_width)
        y_max = min(y_max, image_height)

        ax.set_xlim(x_min, x_max)
        ax.set_ylim(y_min, y_max)
    else:
        ax.set_xlim(0, image_width)
        ax.set_ylim(0, image_height)

    ax.set_title(title)
    ax.set_xlabel("x [px]")
    ax.set_ylabel("y [px]")
    ax.legend()
    ax.set_aspect("equal")

    plt.tight_layout()

    # ---------------------------
    # Speichern
    # ---------------------------
    if save_path is not None:
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=200)
        print(f"  🖼️ Plot gespeichert: {save_path}")

    if show:
        plt.show()
    else:
        plt.close(fig)
        
        
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt


def _set_axes_3d_compact(
    ax,
    x,
    y,
    z,
    xy_margin_factor=0.06,
    z_margin_factor=0.04,
    z_window=None,
    box_aspect=(1.0, 1.0, 0.72),
):
    """
    Setzt die Achsen für einen 3D-Plot so, dass:
    - x und y gleich skaliert bleiben
    - z auf den tatsächlich relevanten Bereich begrenzt werden kann
    - die Plot-Box flacher dargestellt werden kann, ohne die Daten zu verändern

    Parameters
    ----------
    ax : matplotlib 3D axis
    x, y, z : array-like
        Punktkoordinaten
    xy_margin_factor : float
        Relativer Rand für x/y
    z_margin_factor : float
        Relativer Rand für z
    z_window : tuple[float, float] | None
        Falls gesetzt: explizite z-Grenzen (z_min, z_max)
        Beispiel: (0.92, 1.00)
    box_aspect : tuple[float, float, float]
        Relative Box-Darstellung im Bild, z.B. (1, 1, 0.7)
    """
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    z = np.asarray(z, dtype=float)

    x_min, x_max = np.min(x), np.max(x)
    y_min, y_max = np.min(y), np.max(y)

    if z_window is None:
        z_min, z_max = np.min(z), np.max(z)
    else:
        z_min, z_max = z_window

    x_span = x_max - x_min
    y_span = y_max - y_min
    z_span = z_max - z_min

    if x_span == 0:
        x_span = 1e-6
    if y_span == 0:
        y_span = 1e-6
    if z_span == 0:
        z_span = 1e-6

    # x/y gleich behandeln
    xy_span = max(x_span, y_span)
    xy_half = 0.5 * xy_span * (1 + xy_margin_factor)

    x_center = 0.5 * (x_min + x_max)
    y_center = 0.5 * (y_min + y_max)

    z_center = 0.5 * (z_min + z_max)
    z_half = 0.5 * z_span * (1 + z_margin_factor)

    ax.set_xlim(x_center - xy_half, x_center + xy_half)
    ax.set_ylim(y_center - xy_half, y_center + xy_half)
    ax.set_zlim(z_center - z_half, z_center + z_half)

    ax.set_box_aspect(box_aspect)


def plot_triangulated_points_3d(
    triangulated_points: np.ndarray,
    save_path: str | Path | None = None,
    show: bool = True,
    elev: float = 25,
    azim: float = -60,
    invert_z: bool = True,
    save_extra_views: bool = False,
    marker_size: float = 12,
    cmap: str = "viridis",
    z_window: tuple[float, float] | None = None,
    box_aspect: tuple[float, float, float] = (1.0, 1.0, 0.5),
    orthographic: bool = True,
):
    """
    Plottet triangulierte 3D-Punkte in kompakter Form für Berichte.

    Erwartetes Format:
        [idx_x, idx_y, x, y, z, u, v, line_distance]
    oder allgemeiner:
        Spalten 2, 3, 4 enthalten x, y, z.
    """
    triangulated_points = np.asarray(triangulated_points)

    if triangulated_points.ndim != 2 or triangulated_points.shape[1] < 5:
        raise ValueError(
            "triangulated_points muss ein 2D-Array mit mindestens 5 Spalten sein."
        )

    if len(triangulated_points) == 0:
        raise ValueError("triangulated_points ist leer.")

    x = triangulated_points[:, 2].astype(float)
    y = triangulated_points[:, 3].astype(float)
    z = triangulated_points[:, 4].astype(float)

    def _make_plot(local_elev, local_azim, out_path=None):
        fig = plt.figure(figsize=(7.6, 5.3))
        ax = fig.add_subplot(111, projection="3d")

        if orthographic:
            ax.set_proj_type("ortho")

        scatter = ax.scatter(
            x, y, z,
            c=z,
            cmap=cmap,
            s=marker_size,
            depthshade=False,
            edgecolors="none",
            alpha=0.9
        )

        # Position der Hauptachse holen
        pos = ax.get_position()
        
        # Neue Achse für Colorbar (rechts daneben)
        cbar_width = 0.018
        cbar_pad = 0.13
        
        cbar_height = pos.height * 0.75   # kürzer als Plot
        cbar_y = pos.y0 + (pos.height - cbar_height) / 2
        
        cax = fig.add_axes([
            pos.x1 + cbar_pad,   # rechts vom Plot
            cbar_y,              # vertikal zentriert
            cbar_width,
            cbar_height
        ])
        
        cbar = fig.colorbar(scatter, cax=cax)
        cbar.set_label("z in m")

        _set_axes_3d_compact(
            ax,
            x, y, z,
            xy_margin_factor=0.06,
            z_margin_factor=0.03,
            z_window=z_window,
            box_aspect=box_aspect,
        )

        if invert_z:
            ax.invert_zaxis()

        ax.set_xlabel("x in m", labelpad=8)
        ax.set_ylabel("y in m", labelpad=8)
        ax.set_zlabel("z in m", labelpad=1)

        ax.view_init(elev=local_elev, azim=local_azim)

        # Ruhigeres Layout
        ax.grid(True, linestyle=":", linewidth=0.5, alpha=0.35)
        ax.xaxis.pane.fill = False
        ax.yaxis.pane.fill = False
        ax.zaxis.pane.fill = False

        plt.tight_layout()

        if out_path is not None:
            out_path = Path(out_path)
            out_path.parent.mkdir(parents=True, exist_ok=True)
            fig.savefig(out_path, dpi=350, bbox_inches="tight")
            print(f"  🖼️ 3D-Plot gespeichert: {out_path}")

        if show:
            plt.show()
        else:
            plt.close(fig)

    _make_plot(elev, azim, save_path)

    if save_extra_views and save_path is not None:
        save_path = Path(save_path)
        stem = save_path.stem
        suffix = save_path.suffix if save_path.suffix else ".png"
        parent = save_path.parent

        extra_views = [
            (20, -35, parent / f"{stem}_view1{suffix}"),
            (25, -90, parent / f"{stem}_view2{suffix}"),
            (55, -60, parent / f"{stem}_view3{suffix}"),
        ]

        for e, a, p in extra_views:
            _make_plot(e, a, p)
        
def plot_uv_points(
    uv_points: np.ndarray,
    image_width: int,
    image_height: int,
    title: str = "Fitted UV Points",
    save_path: str | Path | None = None,
    show: bool = True,
    annotate_frame_idx: bool = False
):
    import matplotlib.pyplot as plt
    from pathlib import Path
    import numpy as np

    uv_points = np.asarray(uv_points)

    if uv_points.ndim != 2 or uv_points.shape[1] < 3:
        raise ValueError("uv_points muss die Form (n, 3) mit [u, v, frame_idx] haben.")

    u = uv_points[:, 0].astype(float)
    v = uv_points[:, 1].astype(float)
    frame_idx = uv_points[:, 2].astype(int)

    fig, ax = plt.subplots(figsize=(10, 6))

    # y-Achse wie im Bild: oben links Ursprung -> fürs Plotten invertieren
    v_plot = image_height - v

    ax.scatter(u, v_plot, marker="x", s=40, linewidths=1.2, label=f"UV points ({len(uv_points)})")

    if annotate_frame_idx:
        for uu, vv, idx in zip(u, v_plot, frame_idx):
            ax.text(uu + 3, vv + 3, str(idx), fontsize=7)

    ax.set_xlim(0, image_width)
    ax.set_ylim(0, image_height)
    ax.set_aspect("equal")
    ax.set_xlabel("u [px]")
    ax.set_ylabel("v [px] (plot coordinates)")
    ax.set_title(title)
    ax.legend()
    plt.tight_layout()

    if save_path is not None:
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=200)
        print(f"  🖼️ UV-Plot gespeichert: {save_path}")

    if show:
        plt.show()
    else:
        plt.close(fig)