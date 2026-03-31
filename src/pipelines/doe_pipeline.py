from src.config.config import (
    DETECTION_FACTOR,
    DETECTION_THRESHOLD,
    NEIGHBORHOOD_SIZE,
    DISTANCE_FACTOR_MIN,
    DISTANCE_FACTOR_MAX,
    BOUNDARY_FACTOR,
    FIT_METHOD,
    LIMIT_FACTOR,
    THRESHOLD_FACTOR,
    SUBTRACT_BACKGROUND,
    OVERWRITE_RESULTS,
    SHOW_PLOTS,
    SAVE_PLOTS,
    ZOOM_PLOTS,
    Z_MIN_SPAN,
)

from src.io.io_utils import load_or_create_npy_folder, save_result_for_input_folder
from src.detection.detection import detect_laser_points
from src.visualization.plot_utils import (
    plot_image_with_peaks_and_fit,
    plot_triangulated_points_3d,
)
from src.utils.path_utils import get_output_folder_for_input
from src.fitting.fitting import fit_laser_points
from src.utils.lpc_indexing import assign_doe_indices, check_unique_indices
from src.triangulation.triangulation import (
    load_metadata,
    triangulate_indexed_points,
)


def process_single_doe_image(name, arr, input_folder_path, metadata):
    print(f"\n🖼️ Verarbeite Bild: {name}")
    print(f"  Shape: {arr.shape}")
    print(f"  Datentyp: {arr.dtype}")
    print(f"  Intensität: min={arr.min()}, max={arr.max()}")

    peaks = detect_laser_points(
        brightness_array=arr,
        factor=DETECTION_FACTOR,
        threshold=DETECTION_THRESHOLD,
        neighborhood_size=NEIGHBORHOOD_SIZE,
        distance_factor_min=DISTANCE_FACTOR_MIN,
        distance_factor_max=DISTANCE_FACTOR_MAX,
        boundary_factor=BOUNDARY_FACTOR
    )
    print(f"  🔍 Erkannte Peaks: {len(peaks)}")

    fit_results = fit_laser_points(
        brightness_array=arr,
        peaks=peaks,
        method=FIT_METHOD,
        limit_factor=LIMIT_FACTOR,
        threshold_factor=THRESHOLD_FACTOR,
        subtract_background=SUBTRACT_BACKGROUND
    )

    fitted_centers = fit_results["global_centers"]
    print(f"  📍 Gefittete Zentren: {len(fitted_centers)}")

    indexed_lpc = assign_doe_indices(fitted_centers)

    unique_ok = check_unique_indices(indexed_lpc)
    print(f"  🏷️ Eindeutige Indizes: {unique_ok}")

    triangulated_points = triangulate_indexed_points(indexed_lpc, metadata)

    tri_save_path = save_result_for_input_folder(
        triangulated_points,
        input_folder=input_folder_path,
        file_name=f"{name}_triangulated_points",
        overwrite=OVERWRITE_RESULTS
    )
    print(f"  💾 Triangulierte Punkte gespeichert: {tri_save_path}")

    peaks_save_path = save_result_for_input_folder(
        peaks,
        input_folder=input_folder_path,
        file_name=f"{name}_detected_peaks",
        overwrite=OVERWRITE_RESULTS
    )
    print(f"  💾 Peaks gespeichert: {peaks_save_path}")

    fit_save_path = save_result_for_input_folder(
        fitted_centers,
        input_folder=input_folder_path,
        file_name=f"{name}_fitted_centers",
        overwrite=OVERWRITE_RESULTS
    )
    print(f"  💾 Fitted Centers gespeichert: {fit_save_path}")

    indexed_save_path = save_result_for_input_folder(
        indexed_lpc,
        input_folder=input_folder_path,
        file_name=f"{name}_indexed_lpc",
        overwrite=OVERWRITE_RESULTS
    )
    print(f"  💾 Indizierte LPC gespeichert: {indexed_save_path}")

    output_folder = get_output_folder_for_input(input_folder_path)

    if SAVE_PLOTS:
        plot_path = output_folder / f"{name}_detection_fit_plot.png"
        plot_image_with_peaks_and_fit(
            image=arr,
            peaks=peaks,
            fitted_centers=fitted_centers,
            title=f"Detection + Fit: {name}",
            save_path=plot_path,
            show=SHOW_PLOTS,
            zoom=ZOOM_PLOTS
        )

        plot_3d_path = output_folder / f"{name}_triangulated_3d_plot.png"
        plot_triangulated_points_3d(
            triangulated_points=triangulated_points,
            title=f"Triangulated 3D Points: {name}",
            save_path=plot_3d_path,
            show=SHOW_PLOTS,
            z_min_span=Z_MIN_SPAN
        )

    return {
        "name": name,
        "peaks": peaks,
        "fitted_centers": fitted_centers,
        "indexed_lpc": indexed_lpc,
        "triangulated_points": triangulated_points,
    }


def run_doe_folder(folder_name):
    print("🔧 DOE Evaluation gestartet")

    arrays, input_folder_path = load_or_create_npy_folder(folder_name)

    print("\n📂 Eingelesene Bilddaten:")
    print(f"  Anzahl Bilder: {len(arrays)}")
    print(f"  Input-Ordner: {input_folder_path}")

    metadata_path = input_folder_path / "simulation_metadata.json"
    metadata = load_metadata(metadata_path)

    results = {}

    for name, arr in arrays.items():
        results[name] = process_single_doe_image(
            name=name,
            arr=arr,
            input_folder_path=input_folder_path,
            metadata=metadata
        )

    print("\n✅ DOE Evaluation abgeschlossen")
    return results