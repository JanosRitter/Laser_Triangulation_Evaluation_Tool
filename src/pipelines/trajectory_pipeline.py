from __future__ import annotations

import numpy as np

from src.config.config import (
    FIT_METHOD,
    THRESHOLD_FACTOR,
    OVERWRITE_RESULTS,
    SHOW_PLOTS,
    SAVE_PLOTS,
    Z_MIN_SPAN,
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
from src.triangulation.triangulation import (
    triangulate_trajectory_uv_points,
)
from src.visualization.plot_utils import (
    plot_triangulated_points_3d,
    plot_uv_points,
)
from src.utils.path_utils import get_output_folder_for_input


def fit_single_crop(
    crop_array: np.ndarray,
    method: str = "gaussian",
    threshold_factor: float = 2.5,
):
    """
    Fittet genau einen Peak in einem Crop-Bild.
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


def local_crop_center_to_global_uv(local_center: np.ndarray, frame_row: dict) -> np.ndarray:
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


def build_uv_frame_result(frame_row: dict, global_uv: np.ndarray) -> np.ndarray:
    """
    Baut die Zwischenrepräsentation auf:

    [u, v, frame_idx]
    """
    frame_idx = int(frame_row["frame_idx"])
    return np.array([global_uv[0], global_uv[1], frame_idx], dtype=float)


def reformat_trajectory_points_for_plot(triangulated_points_raw: np.ndarray) -> np.ndarray:
    """
    Wandelt trajectory-Triangulation von

        [x, y, z, u, v, frame_idx]

    in ein Format um, das zur bestehenden Plot-Funktion passt:

        [frame_idx, unused, x, y, z, u, v]

    Damit liegen x,y,z in den Spalten 2,3,4.
    """
    triangulated_points_raw = np.asarray(triangulated_points_raw)

    if triangulated_points_raw.ndim != 2 or triangulated_points_raw.shape[1] != 6:
        raise ValueError(
            "triangulated_points_raw muss die Form (n, 6) mit "
            "[x, y, z, u, v, frame_idx] haben."
        )

    x = triangulated_points_raw[:, 0]
    y = triangulated_points_raw[:, 1]
    z = triangulated_points_raw[:, 2]
    u = triangulated_points_raw[:, 3]
    v = triangulated_points_raw[:, 4]
    frame_idx = triangulated_points_raw[:, 5]

    unused = np.full_like(frame_idx, -1)

    reformatted = np.column_stack([
        frame_idx,
        unused,
        x,
        y,
        z,
        u,
        v,
    ]).astype(np.float32)

    return reformatted


def run_trajectory_folder(input_folder: str):
    """
    Trajectory-Auswertung:

    1. run_metadata.json laden
    2. frame_table.csv laden
    3. valide Crop-Frames iterieren
    4. jeden Crop fitten
    5. lokale Fit-Koordinate in globale Bildkoordinate umrechnen
    6. trajectory-Punkte triangulieren
    7. Ergebnis in plot-kompatiblem Format speichern
    8. UV-Plot und 3D-Plot erzeugen
    """
    print("🔧 Trajectory Evaluation gestartet")

    folder_path = resolve_trajectory_input_folder(input_folder)
    run_metadata = load_run_metadata(folder_path)
    frame_table = load_frame_table(folder_path)
    valid_rows = filter_valid_crop_rows(frame_table)

    print("\n📂 Eingelesene Trajectory-Daten:")
    print(f"  Input-Ordner: {folder_path}")
    print(f"  Gesamtframes: {len(frame_table)}")
    print(f"  Valide Crop-Frames: {len(valid_rows)}")
    print(f"  Fit-Methode: {FIT_METHOD}")

    uv_results = []

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
        global_uv = local_crop_center_to_global_uv(local_center, frame_row)

        uv_row = build_uv_frame_result(
            frame_row=frame_row,
            global_uv=global_uv,
        )
        uv_results.append(uv_row)

        print(f"  📍 Lokal gefittet: x={local_center[0]:.2f}, y={local_center[1]:.2f}")
        print(f"  🌍 Global u,v: u={global_uv[0]:.2f}, v={global_uv[1]:.2f}")

    if len(uv_results) == 0:
        uv_results_array = np.empty((0, 3), dtype=float)
    else:
        uv_results_array = np.vstack(uv_results)

    uv_save_path = save_result_for_input_folder(
        uv_results_array,
        input_folder=folder_path,
        file_name="trajectory_fitted_uv_points",
        overwrite=OVERWRITE_RESULTS,
    )
    print(f"\n💾 UV-Ergebnisse gespeichert: {uv_save_path}")

    frame_rows_by_idx = {
        int(row["frame_idx"]): row
        for row in frame_table
    }

    triangulated_points_raw = triangulate_trajectory_uv_points(
        uv_points=uv_results_array,
        frame_rows_by_idx=frame_rows_by_idx,
        metadata=run_metadata,
    )

    # In plot-kompatibles Format umordnen:
    # [frame_idx, -1, x, y, z, u, v]
    triangulated_points = reformat_trajectory_points_for_plot(triangulated_points_raw)

    tri_save_path = save_result_for_input_folder(
        triangulated_points,
        input_folder=folder_path,
        file_name="trajectory_triangulated_points",
        overwrite=OVERWRITE_RESULTS,
    )
    print(f"💾 Triangulierte Punkte gespeichert: {tri_save_path}")

    if SAVE_PLOTS:
        output_folder = get_output_folder_for_input(folder_path)

        # UV-Plot zur Kontrolle gegen preview_sum
        camera = run_metadata["camera"]
        uv_plot_path = output_folder / "trajectory_uv_plot.png"

        plot_uv_points(
            uv_points=uv_results_array,
            image_width=int(camera["img_width"]),
            image_height=int(camera["img_height"]),
            title="Trajectory Fitted UV Points",
            save_path=uv_plot_path,
            show=SHOW_PLOTS,
            annotate_frame_idx=True,
        )

        # 3D-Plot mit bestehender Funktion
        plot_3d_path = output_folder / "trajectory_triangulated_3d_plot.png"

        plot_triangulated_points_3d(
            triangulated_points=triangulated_points,
            title="Trajectory Triangulated 3D Points",
            save_path=plot_3d_path,
            show=SHOW_PLOTS,
            z_min_span=Z_MIN_SPAN,
        )

    print("\n✅ Trajectory Evaluation abgeschlossen")

    return {
        "run_metadata": run_metadata,
        "frame_table": frame_table,
        "valid_rows": valid_rows,
        "uv_results": uv_results_array,
        "triangulated_points_raw": triangulated_points_raw,
        "triangulated_points": triangulated_points,
    }