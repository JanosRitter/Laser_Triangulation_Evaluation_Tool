from src.io.io_utils import load_or_create_npy_folder, save_result_for_input_folder
from src.detection.detection import detect_laser_points
from src.visualization.plot_utils import (
    plot_image_with_peaks_and_fit,
    plot_triangulated_points_3d
)
from src.utils.path_utils import get_output_folder_for_input
from src.fitting.fitting import fit_laser_points
from src.utils.lpc_indexing import assign_doe_indices, check_unique_indices
from src.triangulation.triangulation import (
    load_metadata,
    triangulate_indexed_points
)



def main():
    print("🔧 Test: Input + Detection + Fitting gestartet")

    folder_name = "2026-03-27_14-26-00"

    arrays, input_folder_path = load_or_create_npy_folder(folder_name)

    print("\n📂 Eingelesene Bilddaten:")
    print(f"  Anzahl Bilder: {len(arrays)}")
    print(f"  Input-Ordner: {input_folder_path}")

    for name, arr in arrays.items():
        print(f"\n🖼️ Verarbeite Bild: {name}")
        print(f"  Shape: {arr.shape}")
        print(f"  Datentyp: {arr.dtype}")
        print(f"  Intensität: min={arr.min()}, max={arr.max()}")

        peaks = detect_laser_points(
            brightness_array=arr,
            factor=4,
            threshold=100,
            neighborhood_size=5,
            distance_factor_min=0.2,
            distance_factor_max=2.0,
            boundary_factor=2.5
        )

        print(f"  🔍 Erkannte Peaks: {len(peaks)}")

        fit_results = fit_laser_points(
            brightness_array=arr,
            peaks=peaks,
            method="gaussian",   # oder "threshold_centroid"
            limit_factor=0.35,
            threshold_factor=2.5,
            subtract_background=True
        )

        fitted_centers = fit_results["global_centers"]
        print(f"  📍 Gefittete Zentren: {len(fitted_centers)}")
        
        fitted_centers = fit_results["global_centers"]
        print(f"  📍 Gefittete Zentren: {len(fitted_centers)}")
        
        indexed_lpc = assign_doe_indices(fitted_centers)
        
        
        unique_ok = check_unique_indices(indexed_lpc)
        print(f"  🏷️ Eindeutige Indizes: {unique_ok}")
        
        metadata_path = input_folder_path / f"simulation_metadata.json"
        metadata = load_metadata(metadata_path)
        
        
        triangulated_points = triangulate_indexed_points(indexed_lpc, metadata)
               
        print(triangulated_points[:,[0,1,5,6]])
        
        tri_output_name = f"{name}_triangulated_points"
        tri_save_path = save_result_for_input_folder(
            triangulated_points,
            input_folder=input_folder_path,
            file_name=tri_output_name,
            overwrite=True
        )
        print(f"  💾 Triangulierte Punkte gespeichert: {tri_save_path}")

        # ---------------------------
        # Peaks speichern
        # ---------------------------
        peaks_output_name = f"{name}_detected_peaks"
        peaks_save_path = save_result_for_input_folder(
            peaks,
            input_folder=input_folder_path,
            file_name=peaks_output_name,
            overwrite=True
        )
        print(f"  💾 Peaks gespeichert: {peaks_save_path}")

        # ---------------------------
        # Fitted centers speichern
        # ---------------------------
        fit_output_name = f"{name}_fitted_centers"
        fit_save_path = save_result_for_input_folder(
            fitted_centers,
            input_folder=input_folder_path,
            file_name=fit_output_name,
            overwrite=True
        )
        print(f"  💾 Fitted Centers gespeichert: {fit_save_path}")
        
        indexed_output_name = f"{name}_indexed_lpc"
        indexed_save_path = save_result_for_input_folder(
            indexed_lpc,
            input_folder=input_folder_path,
            file_name=indexed_output_name,
            overwrite=True
        )
        print(f"  💾 Indizierte LPC gespeichert: {indexed_save_path}")

        # ---------------------------
        # Kontrollplot Detection
        # ---------------------------
        output_folder = get_output_folder_for_input(input_folder_path)
        plot_path = output_folder / f"{name}_detection_fit_plot.png"

        plot_image_with_peaks_and_fit(
            image=arr,
            peaks=peaks,
            fitted_centers=fitted_centers,
            title=f"Detection + Fit: {name}",
            save_path=plot_path,
            show=True,
            zoom=True
        )
        
        plot_3d_path = output_folder / f"{name}_triangulated_3d_plot.png"

        plot_triangulated_points_3d(
            triangulated_points=triangulated_points,
            title=f"Triangulated 3D Points: {name}",
            save_path=plot_3d_path,
            show=True,
            z_min_span=0.05
        )

    print("\n✅ Input-, Detection- und Fitting-Test abgeschlossen")


if __name__ == "__main__":
    main()