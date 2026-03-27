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


def plot_triangulated_points_3d(
    triangulated_points: np.ndarray,
    title: str = "Triangulated 3D Points",
    save_path: str | Path | None = None,
    show: bool = True,
    z_min_span: float = 0.05,
    margin_factor: float = 0.1,
    elev: float = 25,
    azim: float = -60
):
    """
    Plottet triangulierte 3D-Punkte als 3D-Scatterplot.

    Erwartetes Format von triangulated_points:
        [idx_x, idx_y, x, y, z, u, v, line_distance]
    oder allgemeiner:
        Spalten 2, 3, 4 enthalten x, y, z.

    Eigenschaften:
    - automatische Achsenskalierung
    - Färbung nach z-Wert
    - Mindestspanne für die z-Achse, damit nahezu ebene Punktmengen
      nicht wie eine zufällige Punktewolke aussehen

    Parameters
    ----------
    triangulated_points : np.ndarray
        Array mit mindestens 5 Spalten, wobei Spalte 2,3,4 = x,y,z.
    title : str
        Plot-Titel.
    save_path : str | Path | None
        Optionaler Speicherpfad.
    show : bool
        Ob der Plot angezeigt werden soll.
    z_min_span : float
        Mindestspanne der z-Achse.
    margin_factor : float
        Relativer Rand um die Daten.
    elev : float
        Elevation des 3D-Plots.
    azim : float
        Azimut des 3D-Plots.
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

    fig = plt.figure(figsize=(9, 7))
    ax = fig.add_subplot(111, projection="3d")

    scatter = ax.scatter(
        x, y, z,
        c=z,
        cmap="viridis",
        s=50,
        depthshade=True
    )

    cbar = fig.colorbar(scatter, ax=ax, pad=0.1)
    cbar.set_label("z")

    # ---------------------------
    # Automatische Achsenskalierung
    # ---------------------------
    x_min, x_max = np.min(x), np.max(x)
    y_min, y_max = np.min(y), np.max(y)
    z_min, z_max = np.min(z), np.max(z)

    x_span = x_max - x_min
    y_span = y_max - y_min
    z_span = z_max - z_min

    # Fallback bei degenerierten Achsen
    if x_span == 0:
        x_span = 1e-6
    if y_span == 0:
        y_span = 1e-6
    if z_span == 0:
        z_span = 1e-6

    x_margin = x_span * margin_factor
    y_margin = y_span * margin_factor

    # Für z eine Mindestspanne erzwingen
    effective_z_span = max(z_span, z_min_span)
    z_margin = effective_z_span * margin_factor

    x_center = 0.5 * (x_min + x_max)
    y_center = 0.5 * (y_min + y_max)
    z_center = 0.5 * (z_min + z_max)

    ax.set_xlim(x_min - x_margin, x_max + x_margin)
    ax.set_ylim(y_min - y_margin, y_max + y_margin)
    ax.set_zlim(
        z_center - 0.5 * effective_z_span - z_margin,
        z_center + 0.5 * effective_z_span + z_margin
    )

    # ---------------------------
    # Achsenbeschriftung / Ansicht
    # ---------------------------
    ax.set_title(title)
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_zlabel("z")

    ax.view_init(elev=elev, azim=azim)

    plt.tight_layout()

    # ---------------------------
    # Speichern
    # ---------------------------
    if save_path is not None:
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=200)
        print(f"  🖼️ 3D-Plot gespeichert: {save_path}")

    if show:
        plt.show()
    else:
        plt.close(fig)