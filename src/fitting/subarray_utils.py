import numpy as np
from scipy.spatial import KDTree


def find_average_peak_distance(peaks: np.ndarray, factor: float = 0.35) -> tuple[int, int]:
    """
    Berechnet den mittleren Abstand der Peaks zum nächsten Nachbarn
    und skaliert daraus ein Fensterlimit.
    """
    if len(peaks) < 2:
        return 10, 10

    kdtree = KDTree(peaks)
    distances, _ = kdtree.query(peaks, k=2)
    avg_distance = np.mean(distances[:, 1])

    limits = (int(avg_distance * factor), int(avg_distance * factor))
    return limits


def create_subarrays(
    brightness_array: np.ndarray,
    peaks: np.ndarray,
    limit_factor: float = 0.35
):
    """
    Schneidet um jeden Peak ein lokales Subarray aus.

    Returns
    -------
    subarrays : list[np.ndarray]
        Liste lokaler 2D-Fenster
    windows : np.ndarray
        Array (n, 4) mit [x_min, x_max, y_min, y_max]
    """
    x_limit, y_limit = find_average_peak_distance(peaks, factor=limit_factor)

    subarrays = []
    windows = []

    for peak in peaks:
        x_center, y_center = peak

        x_min = max(int(x_center - x_limit), 0)
        x_max = min(int(x_center + x_limit), brightness_array.shape[1])
        y_min = max(int(y_center - y_limit), 0)
        y_max = min(int(y_center + y_limit), brightness_array.shape[0])

        sub = brightness_array[y_min:y_max, x_min:x_max]
        subarrays.append(sub)
        windows.append([x_min, x_max, y_min, y_max])

    return subarrays, np.array(windows, dtype=int)


def stack_subarrays_if_possible(subarrays: list[np.ndarray]) -> np.ndarray:
    """
    Wandelt eine Liste gleich großer Subarrays in ein 3D-Array um.
    Falls unterschiedlich groß, wird ein Fehler geworfen.

    shape -> (k, h, w)
    """
    if len(subarrays) == 0:
        return np.empty((0, 0, 0))

    shapes = [sub.shape for sub in subarrays]
    if len(set(shapes)) != 1:
        raise ValueError(
            "Subarrays haben nicht alle dieselbe Größe. "
            "Für das Stapeln müssen alle Fenster identische Shape haben."
        )

    return np.stack(subarrays, axis=0)


def local_to_global_centers(local_centers: np.ndarray, windows: np.ndarray) -> np.ndarray:
    """
    Rechnet lokale Fit-Zentren wieder in globale Bildkoordinaten um.

    Parameters
    ----------
    local_centers : np.ndarray
        Array (n, 2) mit [x_local, y_local]
    windows : np.ndarray
        Array (n, 4) mit [x_min, x_max, y_min, y_max]

    Returns
    -------
    np.ndarray
        Array (n, 2) mit [x_global, y_global]
    """
    global_centers = np.zeros_like(local_centers, dtype=float)

    global_centers[:, 0] = windows[:, 0] + local_centers[:, 0]
    global_centers[:, 1] = windows[:, 2] + local_centers[:, 1]

    return np.round(global_centers, decimals=2)