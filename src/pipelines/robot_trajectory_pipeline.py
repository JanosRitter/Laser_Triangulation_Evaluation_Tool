from __future__ import annotations

import numpy as np

from src.config.config import (
    FIT_METHOD,
    THRESHOLD_FACTOR,
    OVERWRITE_RESULTS,
)

from src.io.io_utils import save_result_for_input_folder
from src.io.ray_io import save_rays
from src.io.trajectory_io import (
    resolve_trajectory_input_folder,
    load_run_metadata,
    load_frame_table,
    filter_valid_crop_rows,
    iter_valid_crop_frames,
)

from src.utils.path_utils import get_output_folder_for_input

from src.fitting.fit_methods import (
    fit_single_slice_gaussian,
    fit_single_slice_threshold_centroid,
)

from src.triangulation.camera_rays import get_camera_ray_from_pixel
from src.triangulation.laser_rays.robot_trajectory_rays import (
    get_robot_trajectory_laser_ray_from_frame_row,
)
from src.triangulation.metadata_io import load_robot_to_camera_calibration
from src.triangulation.laser_triangulation import triangulate_ray_pair

from src.config.config import (
    FIT_METHOD,
    THRESHOLD_FACTOR,
    OVERWRITE_RESULTS,
    SHOW_PLOTS,
    SAVE_PLOTS,
)

from src.visualization.plot_utils import (
    plot_triangulated_points_3d,
    plot_uv_points,
)

from src.visualization.surface_plot import (
    plot_surface_from_triangulated_points,
)

from src.visualization.interactive_point_cloud import (
    save_interactive_point_cloud_html,
)
from src.triangulation.camera_intrinsics import (
    load_camera_intrinsics_for_run,
    camera_ray_from_pixel_with_intrinsics,
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

def reformat_robot_trajectory_points_for_plot(
    triangulated_points_raw: np.ndarray,
) -> np.ndarray:
    """
    Wandelt Robot-Trajectory-Triangulation von

        [x_C, y_C, z_C, u, v, frame_idx, line_distance]

    in plot-kompatibles Format:

        [frame_idx, line_distance, x_C, y_C, z_C, u, v]
    """
    triangulated_points_raw = np.asarray(triangulated_points_raw)

    if triangulated_points_raw.ndim != 2 or triangulated_points_raw.shape[1] != 7:
        raise ValueError(
            "triangulated_points_raw muss die Form (n, 7) mit "
            "[x_C, y_C, z_C, u, v, frame_idx, line_distance] haben."
        )

    x = triangulated_points_raw[:, 0]
    y = triangulated_points_raw[:, 1]
    z = triangulated_points_raw[:, 2]
    u = triangulated_points_raw[:, 3]
    v = triangulated_points_raw[:, 4]
    frame_idx = triangulated_points_raw[:, 5]
    line_distance = triangulated_points_raw[:, 6]

    return np.column_stack([
        frame_idx,
        line_distance,
        x,
        y,
        z,
        u,
        v,
    ]).astype(np.float32)


def run_robot_trajectory_folder(input_folder: str):
    """
    Neue echte Robot-Trajectory-Auswertung.

    Ablauf:
    1. run_metadata.json laden
    2. frame_table.csv laden
    3. robot_to_camera_transform.json laden
    4. valide Crop-Frames iterieren
    5. jeden Crop fitten
    6. lokale Fit-Koordinate in globale Bildkoordinate umrechnen
    7. Kameraray im Kamera-KS rekonstruieren
    8. Laserray aus Roboterpose rekonstruieren und ins Kamera-KS transformieren
    9. Ray-Paare triangulieren
    10. UV-Ergebnisse, Rays und triangulierte Punkte speichern

    Noch nicht enthalten:
    - OpenCV-Kameramodell
    """
    print("🔧 Robot-Trajectory Evaluation gestartet")

    folder_path = resolve_trajectory_input_folder(input_folder)
    run_metadata = load_run_metadata(folder_path)
    frame_table = load_frame_table(folder_path)
    valid_rows = filter_valid_crop_rows(frame_table)

    robot_to_camera_calibration = load_robot_to_camera_calibration(
        run_folder=folder_path,
    )

    print("\n📂 Eingelesene Robot-Trajectory-Daten:")
    print(f"  Input-Ordner: {folder_path}")
    print(f"  Gesamtframes: {len(frame_table)}")
    print(f"  Valide Crop-Frames: {len(valid_rows)}")
    print(f"  Fit-Methode: {FIT_METHOD}")
    
    camera_intrinsics = load_camera_intrinsics_for_run(
        run_folder=folder_path,
        fallback_intrinsics=robot_to_camera_calibration["metadata"]["intrinsics"],
    )
    
    print("\n📷 Kamera-Intrinsics:")
    print(f"  Quelle: {camera_intrinsics['source']}")
    print(f"  fx={camera_intrinsics['fx']}")
    print(f"  fy={camera_intrinsics['fy']}")
    print(f"  cx={camera_intrinsics['cx']}")
    print(f"  cy={camera_intrinsics['cy']}")
    print(f"  dist_coeffs={camera_intrinsics['dist_coeffs']}")

    uv_results = []
    camera_ray_results = []
    laser_ray_results = []
    triangulated_results = []
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
        # Kameraray im Kamera-KS
        # ------------------------------------------------------------
        camera_origin_C = np.array([0.0, 0.0, 0.0], dtype=float)

        camera_direction_C = camera_ray_from_pixel_with_intrinsics(
            u=float(global_uv[0]),
            v=float(global_uv[1]),
            intrinsics=camera_intrinsics,
        )

        # ------------------------------------------------------------
        # Laserray aus Roboterpose, transformiert ins Kamera-KS
        # ------------------------------------------------------------
        try:
            laser_origin_C, laser_direction_C = get_robot_trajectory_laser_ray_from_frame_row(
                frame_row=frame_row,
                calibration=robot_to_camera_calibration,
            )

            laser_ray_results.append(
                {
                    "frame_idx": frame_idx,
                    "origin": laser_origin_C,
                    "direction": laser_direction_C,
                    "uv": global_uv,
                }
            )

            # ------------------------------------------------------------
            # Triangulation im Kamera-KS
            # ------------------------------------------------------------
            point_C, line_distance = triangulate_ray_pair(
                laser_origin=laser_origin_C,
                laser_direction=laser_direction_C,
                camera_origin=camera_origin_C,
                camera_direction=camera_direction_C,
            )

            triangulated_results.append(
                [
                    point_C[0],
                    point_C[1],
                    point_C[2],
                    global_uv[0],
                    global_uv[1],
                    frame_idx,
                    line_distance,
                ]
            )

            print(
                f"  📐 Trianguliert C: "
                f"x={point_C[0]:+.6f}, "
                f"y={point_C[1]:+.6f}, "
                f"z={point_C[2]:+.6f}, "
                f"ray_dist={line_distance * 1000.0:.3f} mm"
            )

        except NotImplementedError as exc:
            ray_build_errors.append(
                {
                    "frame_idx": frame_idx,
                    "message": str(exc),
                }
            )

            if len(ray_build_errors) == 1:
                print(f"  ⚠️ Laser-Ray-Konstruktion noch nicht implementiert: {exc}")

    if len(uv_results) == 0:
        uv_results_array = np.empty((0, 3), dtype=float)
    else:
        uv_results_array = np.vstack(uv_results)

    if len(triangulated_results) == 0:
        triangulated_results_array = np.empty((0, 7), dtype=float)
    else:
        triangulated_results_array = np.asarray(triangulated_results, dtype=float)

    uv_save_path = save_result_for_input_folder(
        uv_results_array,
        input_folder=folder_path,
        file_name="robot_trajectory_fitted_uv_points",
        overwrite=OVERWRITE_RESULTS,
    )

    tri_save_path = save_result_for_input_folder(
        triangulated_results_array,
        input_folder=folder_path,
        file_name="robot_trajectory_triangulated_points_C",
        overwrite=OVERWRITE_RESULTS,
    )

    triangulated_points = reformat_robot_trajectory_points_for_plot(
        triangulated_results_array
    )
    
    tri_plot_save_path = save_result_for_input_folder(
        triangulated_points,
        input_folder=folder_path,
        file_name="robot_trajectory_triangulated_points_C_plot_format",
        overwrite=OVERWRITE_RESULTS,
    )
    
    print(f"\n💾 UV-Ergebnisse gespeichert: {uv_save_path}")
    print(f"💾 Triangulierte Punkte gespeichert: {tri_save_path}")
    print(f"💾 Plot-kompatible Punkte gespeichert: {tri_plot_save_path}")
    
    if SAVE_PLOTS:
        output_folder = get_output_folder_for_input(folder_path)
    
        intrinsics = robot_to_camera_calibration["metadata"]["intrinsics"]
    
        uv_plot_path = output_folder / "robot_trajectory_uv_plot.png"
    
        plot_uv_points(
            uv_points=uv_results_array,
            image_width=int(intrinsics["img_width"]),
            image_height=int(intrinsics["img_height"]),
            title="Robot-Trajectory Fitted UV Points",
            save_path=uv_plot_path,
            show=SHOW_PLOTS,
            annotate_frame_idx=True,
        )
    
        plot_3d_path = output_folder / "robot_trajectory_triangulated_3d_plot_C.png"
    
        plot_triangulated_points_3d(
            triangulated_points=triangulated_points,
            save_path=plot_3d_path,
            show=SHOW_PLOTS,
        )
    
        surface_plot_path = output_folder / "robot_trajectory_surface_plot_C.png"
    
        try:
            plot_surface_from_triangulated_points(
                triangulated_points=triangulated_points,
                save_path=surface_plot_path,
                show=SHOW_PLOTS,
                title="Robot-Trajectory Reconstructed Surface in Camera Frame",
            )
            
            interactive_plot_path = output_folder / "interactive_point_cloud.html"

            save_interactive_point_cloud_html(
                triangulated_points=triangulated_points,
                output_path=interactive_plot_path,
                title="Robot trajectory point cloud",
                annotate_frame_idx=False,
            )
        except ValueError as exc:
            print(f"  ⚠️ Surface-Plot übersprungen: {exc}")

    output_folder = get_output_folder_for_input(folder_path)

    camera_ray_paths = save_rays(
        rays=camera_ray_results,
        output_dir=output_folder,
        stem="robot_trajectory_camera_rays_C",
    )

    laser_ray_paths = save_rays(
        rays=laser_ray_results,
        output_dir=output_folder,
        stem="robot_trajectory_laser_rays_C",
    )

    if ray_build_errors:
        print(
            "\nℹ️ Laser-Ray-Konstruktion wurde vorbereitet, ist aber noch nicht "
            "vollständig implementiert."
        )
        print(f"  Betroffene Frames: {len(ray_build_errors)}")

    print("\n✅ Robot-Trajectory Evaluation abgeschlossen")

    return {
        "run_metadata": run_metadata,
        "frame_table": frame_table,
        "valid_rows": valid_rows,
        "uv_results": uv_results_array,
        "uv_save_path": uv_save_path,
        "camera_ray_results": camera_ray_results,
        "laser_ray_results": laser_ray_results,
        "camera_ray_paths": camera_ray_paths,
        "laser_ray_paths": laser_ray_paths,
        "triangulated_results": triangulated_results_array,
        "triangulated_save_path": tri_save_path,
        "ray_build_errors": ray_build_errors,
        "triangulated_points": triangulated_points,
        "triangulated_plot_save_path": tri_plot_save_path,
    }