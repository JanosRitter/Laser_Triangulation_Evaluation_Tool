from pathlib import Path
import numpy as np


def get_project_root() -> Path:
    """
    Bestimmt den Projekt-Root (laser_triangulation),
    unabhängig davon, von wo das Skript gestartet wird.
    """
    return Path(__file__).resolve().parents[2]
    # src/test -> parents[0]=test, [1]=src, [2]=laser_triangulation


def load_arrays(folder_name: str):
    """
    Lädt Ground Truth (Input) und triangulierte Punkte (Output).
    """
    project_root = get_project_root()

    base_input = project_root / "data" / "input" / "images" / folder_name
    base_output = project_root / "data" / "output" / folder_name

    gt_path = base_input / "ground_truth_points.npy"
    tri_path = base_output / "laser_sim_triangulated_points.npy"

    print("\n🔍 DEBUG PFAD:")
    print(f"  Projekt-Root: {project_root}")
    print(f"  Ground Truth: {gt_path}")
    print(f"  Trianguliert: {tri_path}")

    if not gt_path.exists():
        raise FileNotFoundError(f"Ground-Truth-Datei nicht gefunden: {gt_path}")

    if not tri_path.exists():
        raise FileNotFoundError(f"Triangulations-Datei nicht gefunden: {tri_path}")

    ground_truth = np.load(gt_path)
    triangulated = np.load(tri_path)

    return ground_truth, triangulated, gt_path, tri_path


def check_index_alignment(ground_truth: np.ndarray, triangulated: np.ndarray) -> bool:
    """
    Prüft, ob die Indexspalten [idx_x, idx_y] zeilenweise übereinstimmen.
    """
    gt_idx = ground_truth[:, :2].astype(int)
    tri_idx = triangulated[:, :2].astype(int)

    if gt_idx.shape != tri_idx.shape:
        print("❌ Unterschiedliche Anzahl von Punkten oder Index-Shape.")
        print(f"   Ground Truth: {gt_idx.shape}")
        print(f"   Trianguliert: {tri_idx.shape}")
        return False

    matches = np.all(gt_idx == tri_idx, axis=1)
    if np.all(matches):
        print("✅ Indexzuordnung stimmt zeilenweise überein.")
        return True

    print("❌ Indexzuordnung stimmt NICHT vollständig überein.")
    mismatch_rows = np.where(~matches)[0]

    print(f"   Anzahl fehlerhafter Zeilen: {len(mismatch_rows)}")
    print("   Erste fehlerhafte Zeilen:")
    for row in mismatch_rows[:10]:
        print(
            f"   Zeile {row}: "
            f"GT={tuple(gt_idx[row])}, TRI={tuple(tri_idx[row])}"
        )

    return False


def evaluate_tolerance(
    gt_col: np.ndarray,
    tri_col: np.ndarray,
    rel_tol: float,
    abs_tol: float,
    switch_threshold: float
):
    """
    Vergleicht eine einzelne Spalte robust gegen Werte nahe 0.

    Regeln:
    - Für |GT| < switch_threshold wird nur absolute Toleranz verwendet.
    - Sonst wird relative Toleranz verwendet.
    """
    abs_diff = np.abs(tri_col - gt_col)

    rel_diff = np.full_like(abs_diff, np.nan, dtype=float)
    small_mask = np.abs(gt_col) < switch_threshold
    large_mask = ~small_mask

    rel_diff[large_mask] = abs_diff[large_mask] / np.abs(gt_col[large_mask])

    within_tol = np.zeros_like(abs_diff, dtype=bool)
    within_tol[small_mask] = abs_diff[small_mask] <= abs_tol
    within_tol[large_mask] = rel_diff[large_mask] <= rel_tol

    return within_tol, abs_diff, rel_diff, small_mask


def compare_columns(
    ground_truth: np.ndarray,
    triangulated: np.ndarray,
    rel_tol_xyz: float = 0.02,
    abs_tol_xyz: float = 1e-4,
    switch_threshold_xyz: float = 1e-3,
    rel_tol_uv: float = 0.02,
    abs_tol_uv: float = 0.1,
    switch_threshold_uv: float = 1.0
):
    """
    Vergleicht die Spalten x, y, z, u, v zwischen Ground Truth und Triangulation.

    ground_truth format:
        [idx_x, idx_y, x, y, z, u, v]

    triangulated format:
        [idx_x, idx_y, x, y, z, u, v, line_distance]
    """
    col_names = ["x", "y", "z", "u", "v"]

    gt_vals = ground_truth[:, 2:7]
    tri_vals = triangulated[:, 2:7]

    tol_settings = {
        "x": (rel_tol_xyz, abs_tol_xyz, switch_threshold_xyz),
        "y": (rel_tol_xyz, abs_tol_xyz, switch_threshold_xyz),
        "z": (rel_tol_xyz, abs_tol_xyz, switch_threshold_xyz),
        "u": (rel_tol_uv, abs_tol_uv, switch_threshold_uv),
        "v": (rel_tol_uv, abs_tol_uv, switch_threshold_uv),
    }

    all_within_tol = np.ones_like(gt_vals, dtype=bool)
    all_abs_diff = np.zeros_like(gt_vals, dtype=float)
    all_rel_diff = np.full_like(gt_vals, np.nan, dtype=float)
    all_small_mask = np.zeros_like(gt_vals, dtype=bool)

    print("\n📊 Spaltenvergleich:")
    for i, name in enumerate(col_names):
        rel_tol, abs_tol, switch_threshold = tol_settings[name]

        within_tol, abs_diff, rel_diff, small_mask = evaluate_tolerance(
            gt_col=gt_vals[:, i],
            tri_col=tri_vals[:, i],
            rel_tol=rel_tol,
            abs_tol=abs_tol,
            switch_threshold=switch_threshold
        )

        all_within_tol[:, i] = within_tol
        all_abs_diff[:, i] = abs_diff
        all_rel_diff[:, i] = rel_diff
        all_small_mask[:, i] = small_mask

        col_ok = np.all(within_tol)
        max_abs = np.max(abs_diff) if len(abs_diff) > 0 else 0.0
        mean_abs = np.mean(abs_diff) if len(abs_diff) > 0 else 0.0

        valid_rel = rel_diff[~np.isnan(rel_diff)]
        if len(valid_rel) > 0:
            max_rel = np.max(valid_rel)
            mean_rel = np.mean(valid_rel)
            rel_text = f"mean_rel={100*mean_rel:.3f}%, max_rel={100*max_rel:.3f}%"
        else:
            rel_text = "mean_rel=n/a, max_rel=n/a"

        num_abs_only = int(np.sum(small_mask))

        status = "✅ OK" if col_ok else "❌ FEHLER"
        print(
            f"  {name:>2}: {status} | "
            f"mean_abs={mean_abs:.6g}, max_abs={max_abs:.6g}, "
            f"{rel_text}, abs_only={num_abs_only}"
        )

    all_ok = np.all(all_within_tol)

    failing_rows = np.where(~np.all(all_within_tol, axis=1))[0]
    if len(failing_rows) > 0:
        print("\n⚠️ Zeilen außerhalb der Toleranz:")
        for row in failing_rows[:20]:
            idx_x = int(ground_truth[row, 0])
            idx_y = int(ground_truth[row, 1])

            print(f"\n  Zeile {row} | Index ({idx_x}, {idx_y})")
            for i, name in enumerate(col_names):
                if not all_within_tol[row, i]:
                    gt_val = gt_vals[row, i]
                    tri_val = tri_vals[row, i]
                    abs_diff = all_abs_diff[row, i]
                    rel_diff = all_rel_diff[row, i]
                    used_abs_only = all_small_mask[row, i]

                    if used_abs_only:
                        print(
                            f"    {name}: "
                            f"GT={gt_val:.6g}, TRI={tri_val:.6g}, "
                            f"abs_diff={abs_diff:.6g}, "
                            f"Vergleich=absolute Toleranz"
                        )
                    else:
                        print(
                            f"    {name}: "
                            f"GT={gt_val:.6g}, TRI={tri_val:.6g}, "
                            f"abs_diff={abs_diff:.6g}, "
                            f"rel_diff={100*rel_diff:.3f}%"
                        )

    return all_ok


def compare_line_distance(triangulated: np.ndarray):
    """
    Gibt eine kleine Statistik zur Geraden-Gerade-Distanz aus.
    """
    if triangulated.shape[1] < 8:
        print("\nℹ️ Keine line_distance-Spalte vorhanden.")
        return

    line_distance = triangulated[:, 7]

    print("\n📏 line_distance Statistik:")
    print(f"  mean = {np.mean(line_distance):.6g}")
    print(f"  max  = {np.max(line_distance):.6g}")
    print(f"  min  = {np.min(line_distance):.6g}")


def main():
    folder_name = "2026-03-27_14-26-00"

    rel_tol_xyz = 0.03          # 2%
    abs_tol_xyz = 1e-4
    switch_threshold_xyz = 1e-3

    rel_tol_uv = 0.02           # 2%
    abs_tol_uv = 0.1
    switch_threshold_uv = 1.0

    print("🔧 Vergleich Ground Truth vs. Triangulation gestartet")
    print(f"  Ordner: {folder_name}")
    print(f"  XYZ: rel_tol={100*rel_tol_xyz:.2f}%, abs_tol={abs_tol_xyz:.1e}, switch_threshold={switch_threshold_xyz:.1e}")
    print(f"  UV : rel_tol={100*rel_tol_uv:.2f}%, abs_tol={abs_tol_uv:.3g}, switch_threshold={switch_threshold_uv:.3g}")

    ground_truth, triangulated, gt_path, tri_path = load_arrays(folder_name)

    print("\n📂 Geladene Dateien:")
    print(f"  Ground Truth : {gt_path}")
    print(f"  Trianguliert : {tri_path}")

    print("\n📐 Shapes:")
    print(f"  Ground Truth : {ground_truth.shape}")
    print(f"  Trianguliert : {triangulated.shape}")

    indices_ok = check_index_alignment(ground_truth, triangulated)

    if not indices_ok:
        print("\n🛑 Vergleich abgebrochen, weil die Indexzuordnung nicht stimmt.")
        return

    values_ok = compare_columns(
        ground_truth=ground_truth,
        triangulated=triangulated,
        rel_tol_xyz=rel_tol_xyz,
        abs_tol_xyz=abs_tol_xyz,
        switch_threshold_xyz=switch_threshold_xyz,
        rel_tol_uv=rel_tol_uv,
        abs_tol_uv=abs_tol_uv,
        switch_threshold_uv=switch_threshold_uv
    )

    compare_line_distance(triangulated)

    print("\n🏁 Gesamtergebnis:")
    if values_ok:
        print("✅ Triangulation liegt innerhalb der Toleranz.")
    else:
        print("❌ Triangulation liegt NICHT vollständig innerhalb der Toleranz.")


if __name__ == "__main__":
    main()