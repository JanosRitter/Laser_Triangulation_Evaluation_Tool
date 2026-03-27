import numpy as np
from scipy.spatial import KDTree
from scipy.optimize import minimize
from scipy.ndimage import gaussian_filter


# ============================================================
# PREPROCESSING
# ============================================================
def subtract_mean_background(array: np.ndarray) -> np.ndarray:
    """
    Subtrahiert den Mittelwert eines Arrays und setzt negative Werte auf 0.
    """
    if not isinstance(array, np.ndarray):
        raise ValueError("Input must be a numpy array.")

    adjusted = array.astype(np.float64) - np.mean(array)
    adjusted[adjusted < 0] = 0
    return adjusted


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


# ============================================================
# 2D-GAUSS FIT
# ============================================================
def gaussian_2d_model(xy, mu_x, mu_y, sigma_x, sigma_y, amplitude):
    x, y = xy
    return amplitude * np.exp(
        -(((x - mu_x) ** 2) / (2 * sigma_x ** 2) +
          ((y - mu_y) ** 2) / (2 * sigma_y ** 2))
    )


def gaussian_2d_residuals(params, xy, slice_2d):
    mu_x, mu_y, sigma_x, sigma_y, amplitude = params
    model = gaussian_2d_model(xy, mu_x, mu_y, sigma_x, sigma_y, amplitude)
    residuals = (model - slice_2d.ravel()) ** 2
    weights = slice_2d.ravel() + 1e-6
    return np.sum(residuals * weights)


def prepare_meshgrid(width, height):
    x = np.arange(width)
    y = np.arange(height)
    return np.meshgrid(x, y)


def fit_single_slice_gaussian(slice_2d: np.ndarray):
    """
    Fit einer 2D-Gaußfunktion auf ein einzelnes Subarray.

    Returns
    -------
    center : np.ndarray
        [mu_x, mu_y]
    deviations : np.ndarray
        [sigma_x, sigma_y]
    amplitude : float
    fitted_slice : np.ndarray
    """
    if not isinstance(slice_2d, np.ndarray):
        raise ValueError("Input must be a numpy array.")

    slice_2d = np.asarray(slice_2d, dtype=np.float64)
    height, width = slice_2d.shape

    smoothed = gaussian_filter(slice_2d, sigma=1.5)
    y_max, x_max = np.unravel_index(np.argmax(smoothed), smoothed.shape)

    initial_guess = [x_max, y_max, max(width / 8, 1), max(height / 8, 1), smoothed.max()]
    bounds = [
        (0, width),
        (0, height),
        (1, width),
        (1, height),
        (0.1, None)
    ]

    x_value, y_value = prepare_meshgrid(width, height)
    xy_grid = np.vstack([x_value.ravel(), y_value.ravel()])

    result = minimize(
        gaussian_2d_residuals,
        initial_guess,
        args=(xy_grid, slice_2d),
        bounds=bounds,
        method="L-BFGS-B"
    )

    if not result.success:
        return (
            np.array([np.nan, np.nan]),
            np.array([np.nan, np.nan]),
            np.nan,
            np.full((height, width), np.nan)
        )

    mu_x, mu_y, sigma_x, sigma_y, amplitude = result.x
    fitted = gaussian_2d_model(
        xy_grid, mu_x, mu_y, sigma_x, sigma_y, amplitude
    ).reshape(height, width)

    return (
        np.array([mu_x, mu_y]),
        np.array([sigma_x, sigma_y]),
        amplitude,
        fitted
    )


def fit_gaussian_batch(subarrays_3d: np.ndarray):
    """
    Gaußfit für alle Slices eines 3D-Arrays.

    Returns
    -------
    centers : np.ndarray
        (n, 2)
    deviations : np.ndarray
        (n, 2)
    amplitudes : np.ndarray
        (n,)
    fitted_data : np.ndarray
        (n, h, w)
    """
    if subarrays_3d.ndim != 3:
        raise ValueError("subarrays_3d must be a 3D array of shape (n, h, w).")

    num_slices, height, width = subarrays_3d.shape

    centers = np.zeros((num_slices, 2), dtype=float)
    deviations = np.zeros((num_slices, 2), dtype=float)
    amplitudes = np.zeros(num_slices, dtype=float)
    fitted_data = np.zeros((num_slices, height, width), dtype=float)

    for i in range(num_slices):
        center, dev, amp, fitted = fit_single_slice_gaussian(subarrays_3d[i])
        centers[i] = center
        deviations[i] = dev
        amplitudes[i] = amp
        fitted_data[i] = fitted

    return centers, deviations, amplitudes, fitted_data


# ============================================================
# THRESHOLD-CENTROID FIT
# ============================================================
def fit_single_slice_threshold_centroid(
    slice_2d: np.ndarray,
    threshold_factor: float = 2.5
):
    """
    Schwerpunkt-Fit nach Thresholding.

    Returns
    -------
    center : np.ndarray
        [x_center, y_center]
    uncertainties : np.ndarray
        [sigma_x, sigma_y]
    filtered_slice : np.ndarray
    """
    if not isinstance(slice_2d, np.ndarray):
        raise ValueError("Input must be a numpy array.")

    slice_2d = np.asarray(slice_2d, dtype=np.float64)
    height, width = slice_2d.shape

    mean_val = np.mean(slice_2d)
    std_val = np.std(slice_2d)
    threshold = mean_val + threshold_factor * std_val

    filtered_slice = np.where(slice_2d >= threshold, slice_2d, 0)

    total_mass = np.sum(filtered_slice)
    if total_mass == 0:
        return (
            np.array([np.nan, np.nan]),
            np.array([np.nan, np.nan]),
            filtered_slice
        )

    x_indices, y_indices = np.meshgrid(np.arange(width), np.arange(height))

    x_center = np.sum(x_indices * filtered_slice) / total_mass
    y_center = np.sum(y_indices * filtered_slice) / total_mass

    x_uncertainty = np.sqrt(np.sum((x_indices - x_center) ** 2 * filtered_slice) / total_mass)
    y_uncertainty = np.sqrt(np.sum((y_indices - y_center) ** 2 * filtered_slice) / total_mass)

    return (
        np.array([x_center, y_center]),
        np.array([x_uncertainty, y_uncertainty]),
        filtered_slice
    )


def fit_threshold_centroid_batch(
    subarrays_3d: np.ndarray,
    threshold_factor: float = 2.5
):
    """
    Threshold-Centroid-Fit für alle Slices eines 3D-Arrays.

    Returns
    -------
    centers : np.ndarray
        (n, 2)
    uncertainties : np.ndarray
        (n, 2)
    filtered_data : np.ndarray
        (n, h, w)
    """
    if subarrays_3d.ndim != 3:
        raise ValueError("subarrays_3d must be a 3D array of shape (n, h, w).")

    num_slices, height, width = subarrays_3d.shape

    centers = np.zeros((num_slices, 2), dtype=float)
    uncertainties = np.zeros((num_slices, 2), dtype=float)
    filtered_data = np.zeros((num_slices, height, width), dtype=float)

    for i in range(num_slices):
        center, unc, filtered = fit_single_slice_threshold_centroid(
            subarrays_3d[i],
            threshold_factor=threshold_factor
        )
        centers[i] = center
        uncertainties[i] = unc
        filtered_data[i] = filtered

    return centers, uncertainties, filtered_data


# ============================================================
# HIGH-LEVEL PIPELINE
# ============================================================
def fit_laser_points(
    brightness_array: np.ndarray,
    peaks: np.ndarray,
    method: str = "gaussian",
    limit_factor: float = 0.35,
    threshold_factor: float = 2.5,
    subtract_background: bool = False
):
    """
    Kompletter Fit-Pipeline-Schritt:
    1. Subarrays um Peaks erzeugen
    2. optional Hintergrund abziehen
    3. gewählte Fitting-Methode anwenden
    4. lokale Zentren in globale Koordinaten umrechnen

    Parameters
    ----------
    brightness_array : np.ndarray
    peaks : np.ndarray
    method : str
        "gaussian" oder "threshold_centroid"
    limit_factor : float
        Fenstergröße relativ zum mittleren Peakabstand
    threshold_factor : float
        Nur relevant für threshold_centroid
    subtract_background : bool
        Mittelwert pro Subarray abziehen

    Returns
    -------
    results : dict
        Enthält u. a.:
        - "local_centers"
        - "global_centers"
        - "windows"
        - "subarrays"
        - methodenspezifische Zusatzdaten
    """
    subarrays, windows = create_subarrays(
        brightness_array=brightness_array,
        peaks=peaks,
        limit_factor=limit_factor
    )

    if subtract_background:
        subarrays = [subtract_mean_background(sub) for sub in subarrays]

    subarrays_3d = stack_subarrays_if_possible(subarrays)

    if method == "gaussian":
        local_centers, deviations, amplitudes, fitted_data = fit_gaussian_batch(subarrays_3d)

        return {
            "method": method,
            "subarrays": subarrays_3d,
            "windows": windows,
            "local_centers": local_centers,
            "global_centers": local_to_global_centers(local_centers, windows),
            "deviations": deviations,
            "amplitudes": amplitudes,
            "fitted_data": fitted_data
        }

    elif method == "threshold_centroid":
        local_centers, uncertainties, filtered_data = fit_threshold_centroid_batch(
            subarrays_3d,
            threshold_factor=threshold_factor
        )

        return {
            "method": method,
            "subarrays": subarrays_3d,
            "windows": windows,
            "local_centers": local_centers,
            "global_centers": local_to_global_centers(local_centers, windows),
            "uncertainties": uncertainties,
            "filtered_data": filtered_data
        }

    else:
        raise ValueError(f"Unbekannte Methode: {method}")