from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt


BASE = Path(
    r"C:\Users\JRI\Documents\Robotercode_new\laser_triangulation\data"
    r"\20260520_100303_robot_measurement\Back_ups"
)

PATH_WITH = BASE / "Mit_Kalibrierung" / "robot_trajectory_triangulated_points_C_plot_format.npy"
PATH_WITHOUT = BASE / "Ohne_Kalibrierung" / "robot_trajectory_triangulated_points_C_plot_format.npy"

OUT = BASE / "Vergleich_Kamera_Kalibrierung"
OUT.mkdir(parents=True, exist_ok=True)


def load_plot_format(path: Path):
    """
    Erwartet:
        [frame_idx, line_distance/unused, x, y, z, u, v]
    """
    arr = np.load(path)

    frame_idx = arr[:, 0].astype(int)
    xyz = arr[:, 2:5].astype(float)
    uv = arr[:, 5:7].astype(float)

    return frame_idx, xyz, uv


idx_with, xyz_with, uv_with = load_plot_format(PATH_WITH)
idx_without, xyz_without, uv_without = load_plot_format(PATH_WITHOUT)

if not np.array_equal(idx_with, idx_without):
    raise ValueError("Frame-Indizes stimmen nicht überein.")

delta = xyz_with - xyz_without
delta_norm = np.linalg.norm(delta, axis=1)

print("\nVergleich Mit vs. Ohne Intrinsics")
print(f"Punkte: {len(delta_norm)}")
print(f"Mittlere 3D-Abweichung: {np.mean(delta_norm) * 1000:.4f} mm")
print(f"Median 3D-Abweichung:   {np.median(delta_norm) * 1000:.4f} mm")
print(f"Max 3D-Abweichung:      {np.max(delta_norm) * 1000:.4f} mm")
print()
print(f"Mean dx: {np.mean(delta[:,0]) * 1000:+.4f} mm")
print(f"Mean dy: {np.mean(delta[:,1]) * 1000:+.4f} mm")
print(f"Mean dz: {np.mean(delta[:,2]) * 1000:+.4f} mm")
print(f"Std  dx: {np.std(delta[:,0]) * 1000:.4f} mm")
print(f"Std  dy: {np.std(delta[:,1]) * 1000:.4f} mm")
print(f"Std  dz: {np.std(delta[:,2]) * 1000:.4f} mm")


# ------------------------------------------------------------
# 1) Histogramm der 3D-Abweichung
# ------------------------------------------------------------
fig, ax = plt.subplots(figsize=(8, 5))
ax.hist(delta_norm * 1000, bins=40)
ax.set_title("3D-Abweichung: Mit Kalibrierung - Ohne Kalibrierung")
ax.set_xlabel("Abweichung [mm]")
ax.set_ylabel("Anzahl Punkte")
ax.grid(True)
fig.tight_layout()
fig.savefig(OUT / "01_hist_3d_difference_mm.png", dpi=200)
plt.close(fig)


# ------------------------------------------------------------
# 2) dz über x/y als Draufsicht
# ------------------------------------------------------------
fig, ax = plt.subplots(figsize=(8, 7))
sc = ax.scatter(
    xyz_without[:, 0],
    xyz_without[:, 1],
    c=delta[:, 2] * 1000,
    s=20,
)
ax.set_title("Z-Differenz durch Intrinsics-Kalibrierung")
ax.set_xlabel("x ohne Kalibrierung [m]")
ax.set_ylabel("y ohne Kalibrierung [m]")
ax.set_aspect("equal", adjustable="box")
ax.grid(True)
cbar = fig.colorbar(sc, ax=ax)
cbar.set_label("dz [mm]")
fig.tight_layout()
fig.savefig(OUT / "02_topview_dz_difference_mm.png", dpi=200)
plt.close(fig)


# ------------------------------------------------------------
# 3) 2D-Vektorfeld: laterale Verschiebung x/y
# ------------------------------------------------------------
fig, ax = plt.subplots(figsize=(8, 7))
ax.quiver(
    xyz_without[:, 0],
    xyz_without[:, 1],
    delta[:, 0],
    delta[:, 1],
    angles="xy",
    scale_units="xy",
    scale=1,
)
ax.set_title("Laterale Verschiebung durch Intrinsics-Kalibrierung")
ax.set_xlabel("x ohne Kalibrierung [m]")
ax.set_ylabel("y ohne Kalibrierung [m]")
ax.set_aspect("equal", adjustable="box")
ax.grid(True)
fig.tight_layout()
fig.savefig(OUT / "03_topview_xy_shift_vectors.png", dpi=200)
plt.close(fig)


# ------------------------------------------------------------
# 4) z-Vergleich entlang Frame-Index
# ------------------------------------------------------------
fig, ax = plt.subplots(figsize=(10, 5))
ax.plot(idx_with, xyz_without[:, 2] * 1000, label="ohne Kalibrierung")
ax.plot(idx_with, xyz_with[:, 2] * 1000, label="mit Kalibrierung")
ax.set_title("Z-Verlauf entlang Frame-Index")
ax.set_xlabel("frame_idx")
ax.set_ylabel("z [mm]")
ax.grid(True)
ax.legend()
fig.tight_layout()
fig.savefig(OUT / "04_z_over_frame_comparison.png", dpi=200)
plt.close(fig)


# ------------------------------------------------------------
# 5) Differenz je Achse über Frame-Index
# ------------------------------------------------------------
fig, ax = plt.subplots(figsize=(10, 5))
ax.plot(idx_with, delta[:, 0] * 1000, label="dx")
ax.plot(idx_with, delta[:, 1] * 1000, label="dy")
ax.plot(idx_with, delta[:, 2] * 1000, label="dz")
ax.set_title("Differenz mit - ohne Kalibrierung über Frame-Index")
ax.set_xlabel("frame_idx")
ax.set_ylabel("Differenz [mm]")
ax.grid(True)
ax.legend()
fig.tight_layout()
fig.savefig(OUT / "05_axis_difference_over_frame_mm.png", dpi=200)
plt.close(fig)

print(f"\nPlots gespeichert in:\n{OUT}")