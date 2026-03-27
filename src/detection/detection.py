import numpy as np
from scipy.spatial import KDTree


# ============================================================
# HILFSFUNKTIONEN
# ============================================================
def closest_divisor(shape: int, factor: int) -> int:
    """
    Findet den Divisor von 'shape', der am nächsten an 'factor' liegt.
    """
    divisors = [i for i in range(1, shape + 1) if shape % i == 0]
    return min(divisors, key=lambda x: abs(x - factor))


def block_average(brightness_array: np.ndarray, factor: int = 15):
    """
    Mittelt ein 2D-Bild blockweise herunter.

    Falls factor kein Teiler der Bilddimensionen ist, wird der nächste passende
    Divisor verwendet.

    Returns
    -------
    reduced_array : np.ndarray
        Heruntergerechnetes Bild
    used_factor : int
        Tatsächlich verwendeter Blockfaktor
    """
    shape_y, shape_x = brightness_array.shape

    factor_y = closest_divisor(shape_y, factor)
    factor_x = closest_divisor(shape_x, factor)
    used_factor = min(factor_x, factor_y)

    new_shape = (
        brightness_array.shape[0] // used_factor,
        brightness_array.shape[1] // used_factor
    )

    truncated = brightness_array[
        :new_shape[0] * used_factor,
        :new_shape[1] * used_factor
    ]

    reshaped = truncated.reshape(
        new_shape[0], used_factor,
        new_shape[1], used_factor
    )

    reduced_array = reshaped.mean(axis=(1, 3))
    return reduced_array, used_factor


# ============================================================
# PEAK-DETEKTION
# ============================================================
def detect_peak_candidates(
    brightness_array: np.ndarray,
    factor: int = 15,
    threshold: float | None = None,
    neighborhood_size: int = 5
) -> np.ndarray:
    """
    Findet Peak-Kandidaten in einem blockgemittelten Bild.

    Vorgehen:
    1. Bild blockweise mitteln
    2. Lokale Maxima in einer neighborhood_size x neighborhood_size Umgebung suchen
    3. Peaks wieder auf Originalkoordinaten hochskalieren

    Parameters
    ----------
    brightness_array : np.ndarray
        2D Intensitätsbild
    factor : int
        Blockgröße für das Heruntermitteln
    threshold : float | None
        Minimalwert im reduzierten Bild. Wenn None, wird der Mittelwert verwendet.
    neighborhood_size : int
        Größe des lokalen Suchfensters, bevorzugt ungerade (z. B. 3, 5, 7)

    Returns
    -------
    peaks : np.ndarray
        Array der Form (n, 2) mit Peak-Kandidaten in Originalpixelkoordinaten:
        [x, y]
    """
    if neighborhood_size % 2 == 0:
        raise ValueError("neighborhood_size muss ungerade sein.")

    reduced, used_factor = block_average(brightness_array, factor)

    if threshold is None:
        threshold = np.mean(reduced)

    half = neighborhood_size // 2
    rows, cols = reduced.shape

    peaks = []

    for y in range(half, rows - half):
        for x in range(half, cols - half):
            sub = reduced[y - half:y + half + 1, x - half:x + half + 1]
            center_val = reduced[y, x]

            if center_val == np.max(sub) and center_val > threshold:
                peaks.append([x * used_factor, y * used_factor])

    if len(peaks) == 0:
        return np.empty((0, 2), dtype=int)

    return np.array(peaks, dtype=int)


# ============================================================
# FILTER
# ============================================================
def filter_by_relative_distance(
    points: np.ndarray,
    distance_factor_min: float = 0.2,
    distance_factor_max: float = 2.0
) -> np.ndarray:
    """
    Filtert Punkte anhand ihrer Nachbarabstände.

    Punkte werden entfernt, wenn sie:
    - zu nah an anderen Punkten liegen
    - zu weit von ihrer lokalen Nachbarschaft entfernt liegen
    """
    if len(points) < 3:
        return points

    tree = KDTree(points)
    distances, indices = tree.query(points, k=3)

    nearest_distances = distances[:, 1]
    median_distance = np.median(nearest_distances)

    min_distance = distance_factor_min * median_distance
    max_distance = distance_factor_max * median_distance

    points_to_remove = set()

    for i in range(len(points)):
        d1, d2 = distances[i, 1], distances[i, 2]
        n1 = indices[i, 1]

        if d1 < min_distance or d2 < min_distance:
            points_to_remove.add(max(i, n1))
        elif d1 > max_distance or d2 > max_distance:
            points_to_remove.add(i)

    filtered_points = np.array(
        [p for i, p in enumerate(points) if i not in points_to_remove]
    )

    if len(filtered_points) == 0:
        return np.empty((0, 2), dtype=int)

    return filtered_points


def filter_by_region(points: np.ndarray, boundary_factor: float = 2.5) -> np.ndarray:
    """
    Filtert Punkte anhand einer groben räumlichen Region um den Schwerpunkt.
    """
    if len(points) == 0:
        return points

    center = np.mean(points, axis=0)
    std_dev = np.std(points, axis=0)

    bounds = [
        (center[i] - boundary_factor * std_dev[i],
         center[i] + boundary_factor * std_dev[i])
        for i in range(2)
    ]

    in_bounds = (
        (bounds[0][0] <= points[:, 0]) & (points[:, 0] <= bounds[0][1]) &
        (bounds[1][0] <= points[:, 1]) & (points[:, 1] <= bounds[1][1])
    )

    return points[in_bounds]


def filter_peak_candidates(
    points: np.ndarray,
    distance_factor_min: float = 0.2,
    distance_factor_max: float = 2.0,
    boundary_factor: float = 2.5
) -> np.ndarray:
    """
    Kombiniert Distanz- und Regionenfilter.
    """
    filtered = filter_by_relative_distance(
        points,
        distance_factor_min=distance_factor_min,
        distance_factor_max=distance_factor_max
    )

    filtered = filter_by_region(
        filtered,
        boundary_factor=boundary_factor
    )

    return filtered


# ============================================================
# LOKALE AUSSCHNITTE FÜR FITTING
# ============================================================
def find_average_distance(peaks: np.ndarray, factor: float = 0.5):
    """
    Bestimmt aus den Peak-Abständen ein sinnvolles Fenster für lokale Ausschnitte.
    """
    if len(peaks) < 2:
        return 10, 10

    kdtree = KDTree(peaks)
    distances, _ = kdtree.query(peaks, k=2)
    avg_distance = np.mean(distances[:, 1])

    limits = (int(avg_distance * factor), int(avg_distance * factor))
    return limits


def create_brightness_subarrays(brightness_array: np.ndarray, peaks: np.ndarray):
    """
    Erstellt für jeden Peak einen lokalen Bildausschnitt.

    Returns
    -------
    subarrays : list[np.ndarray]
        Liste lokaler 2D-Ausschnitte
    windows : np.ndarray
        Array der Form (n, 4) mit [x_min, x_max, y_min, y_max]
    """
    x_limit, y_limit = find_average_distance(peaks)

    subarrays = []
    windows = []

    for peak in peaks:
        x_center, y_center = peak

        x_min = max(x_center - x_limit, 0)
        x_max = min(x_center + x_limit, brightness_array.shape[1])
        y_min = max(y_center - y_limit, 0)
        y_max = min(y_center + y_limit, brightness_array.shape[0])

        sub = brightness_array[y_min:y_max, x_min:x_max]
        subarrays.append(sub)
        windows.append([x_min, x_max, y_min, y_max])

    return subarrays, np.array(windows, dtype=int)


# ============================================================
# RÜCKRECHNUNG DER FIT-CENTER
# ============================================================
def local_to_global_centers(local_centers: np.ndarray, peaks: np.ndarray) -> np.ndarray:
    """
    Rechnet lokal bestimmte Subpixel-Zentren wieder in globale Bildkoordinaten um.

    local_centers:
        Array (n, 2) mit lokalen Koordinaten innerhalb der Subarrays

    peaks:
        Array (n, 2) mit groben Peakpositionen im Originalbild
    """
    x_limit, y_limit = find_average_distance(peaks)

    global_centers = np.zeros_like(local_centers, dtype=float)
    global_centers[:, 0] = peaks[:, 0] + local_centers[:, 0] - x_limit
    global_centers[:, 1] = peaks[:, 1] + local_centers[:, 1] - y_limit

    return np.round(global_centers, decimals=1)


# ============================================================
# KOMFORTFUNKTION FÜR SCHRITT 1
# ============================================================
def detect_laser_points(
    brightness_array: np.ndarray,
    factor: int = 15,
    threshold: float | None = None,
    neighborhood_size: int = 5,
    distance_factor_min: float = 0.2,
    distance_factor_max: float = 2.0,
    boundary_factor: float = 2.5
) -> np.ndarray:
    """
    Kompletter Detection-Schritt:
    - block average
    - Peak-Kandidaten
    - Filterung

    Returns
    -------
    peaks : np.ndarray
        Array (n, 2) mit groben Peakpositionen [x, y]
    """
    peaks = detect_peak_candidates(
        brightness_array=brightness_array,
        factor=factor,
        threshold=threshold,
        neighborhood_size=neighborhood_size
    )

    peaks = filter_peak_candidates(
        peaks,
        distance_factor_min=distance_factor_min,
        distance_factor_max=distance_factor_max,
        boundary_factor=boundary_factor
    )

    return peaks