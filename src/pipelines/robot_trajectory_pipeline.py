from __future__ import annotations

import numpy as np

from src.config.config import (
    FIT_METHOD,
    THRESHOLD_FACTOR,
    OVERWRITE_RESULTS,
)

from src.io.io_utils import save_result_for_input_folder
from src.io.trajectory_io import (
    resolve_trajectory_input_folder,
    load_run_metadata,
    load_frame_table,
    filter_valid_crop_rows,
    iter_valid_crop_frames,
)

from src.fitting.fit_methods import (
    fit_single_slice_gaussian,
    fit_single_slice_threshold_centroid,
)

from src.triangulation.laser_rays.robot_trajectory_rays import (
    get_robot_trajectory_laser_ray_from_frame_row,
)


def fit_single_crop(
    crop_array: np.ndarray,
    method: str = "gaussian",
    threshold_factor: float = 2.5,
) -> dict:
    """
    Fittet genau einen Laserpunkt in einem Crop-Bild.
    """
    if method == "gaussian":
        center, deviations, amplitude, fitted = fit_single_slice_gaussian(crop_array)
        return {
            "method": method,
            "local_center": center,
            "deviations": deviations,
            "amplitude": amplitude,
            "fitted_or_filtered": fitted,
        }

    if method == "threshold_centroid":
        center, uncertainties, filtered = fit_single_slice_threshold_centroid(
            crop_array,
            threshold_factor=threshold_factor,
        )
        return {
            "method": method,
            "local_center": center,
            "uncertainties": uncertainties,
            "fitted_or_filtered": filtered,
        }

    raise ValueError(f"Unbekannte Fit-Methode: {method}")


def local_crop_center_to_global_uv(
    local_center: np.ndarray,
    frame_row: dict,
) -> np.ndarray:
    """
    Rechnet lokale Crop-Koordinaten in globale Vollbildkoordinaten um.

    u = crop_x0 + local_x
    v = crop_y0 + local_y
    """
    crop_x0 = float(frame_row["crop_x0"])
    crop_y0 = float(frame_row["crop_y0"])

    u = crop_x0 + float(local_center[0])
    v = crop_y0 + float(local_center[1])

    return np.array([u, v], dtype=float)


def build_robot_trajectory_uv_result(
    frame_row: dict,
    global_uv: np.ndarray,
) -> np.ndarray:
    """
    Zwischenrepräsentation:

    [u, v, frame_idx]
    """
    frame_idx = int(frame_row["frame_idx"])
    return np.array([global_uv[0], global_uv[1], frame_idx], dtype=float)


def run_robot_trajectory_folder(input_folder: str):
    """
    Neue echte Robot-Trajectory-Auswertung.

    Aktueller Stand:
    1. run_metadata.json laden
    2. frame_table.csv laden
    3. valide Crop-Frames iterieren
    4. jeden Crop fitten
    5. lokale Fit-Koordinate in globale Bildkoordinate umrechnen
    6. UV-Ergebnisse speichern
    7. vorbereitend pro Frame die neue Robot-Trajectory-Ray-Funktion aufrufen

    Noch nicht enthalten:
    - OpenCV-Kameramodell
    - robot_to_camera_transform.json
    - echte Laser-Ray-Konstruktion
    - finale Triangulation
    """
    print("🔧 Robot-Trajectory Evaluation gestartet")

    folder_path = resolve_trajectory_input_folder(input_folder)
    run_metadata = load_run_metadata(folder_path)
    frame_table = load_frame_table(folder_path)
    valid_rows = filter_valid_crop_rows(frame_table)

    print("\n📂 Eingelesene Robot-Trajectory-Daten:")
    print(f"  Input-Ordner: {folder_path}")
    print(f"  Gesamtframes: {len(frame_table)}")
    print(f"  Valide Crop-Frames: {len(valid_rows)}")
    print(f"  Fit-Methode: {FIT_METHOD}")

    uv_results = []
    ray_build_errors = []

    for frame_row, crop_array in iter_valid_crop_frames(folder_path):
        frame_idx = int(frame_row["frame_idx"])

        print(f"\n🖼️ Verarbeite Crop-Frame: {frame_idx:06d}")
        print(f"  Crop-Shape: {crop_array.shape}")
        print(f"  Intensität: min={crop_array.min()}, max={crop_array.max()}")

        fit_result = fit_single_crop(
            crop_array=crop_array,
            method=FIT_METHOD,
            threshold_factor=THRESHOLD_FACTOR,
        )

        local_center = fit_result["local_center"]
        global_uv = local_crop_center_to_global_uv(
            local_center=local_center,
            frame_row=frame_row,
        )

        uv_row = build_robot_trajectory_uv_result(
            frame_row=frame_row,
            global_uv=global_uv,
        )
        uv_results.append(uv_row)

        print(f"  📍 Lokal gefittet: x={local_center[0]:.2f}, y={local_center[1]:.2f}")
        print(f"  🌍 Global u,v: u={global_uv[0]:.2f}, v={global_uv[1]:.2f}")

        # ------------------------------------------------------------
        # Platzhalter für neue Laser-Ray-Konstruktion
        # ------------------------------------------------------------
        try:
            _laser_origin, _laser_direction = get_robot_trajectory_laser_ray_from_frame_row(
                frame_row=frame_row,
                T_C_R=None,
            )
        except NotImplementedError as exc:
            ray_build_errors.append(
                {
                    "frame_idx": frame_idx,
                    "message": str(exc),
                }
            )

            # Nicht pro Frame endlos laut werden
            if len(ray_build_errors) == 1:
                print(f"  ⚠️ Laser-Ray-Konstruktion noch nicht implementiert: {exc}")

    if len(uv_results) == 0:
        uv_results_array = np.empty((0, 3), dtype=float)
    else:
        uv_results_array = np.vstack(uv_results)

    uv_save_path = save_result_for_input_folder(
        uv_results_array,
        input_folder=folder_path,
        file_name="robot_trajectory_fitted_uv_points",
        overwrite=OVERWRITE_RESULTS,
    )

    print(f"\n💾 UV-Ergebnisse gespeichert: {uv_save_path}")

    if ray_build_errors:
        print(
            "\nℹ️ Laser-Ray-Konstruktion wurde vorbereitet, ist aber noch nicht "
            "implementiert."
        )
        print(f"  Betroffene Frames: {len(ray_build_errors)}")

    print("\n✅ Robot-Trajectory Evaluation abgeschlossen")

    return {
        "run_metadata": run_metadata,
        "frame_table": frame_table,
        "valid_rows": valid_rows,
        "uv_results": uv_results_array,
        "ray_build_errors": ray_build_errors,
    }