from .preprocessing import subtract_mean_background
from .subarray_utils import (
    create_subarrays,
    stack_subarrays_if_possible,
    local_to_global_centers
)
from .fit_methods import (
    fit_gaussian_batch,
    fit_threshold_centroid_batch
)


def fit_laser_points(
    brightness_array,
    peaks,
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

    if method == "threshold_centroid":
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

    raise ValueError(f"Unbekannte Methode: {method}")