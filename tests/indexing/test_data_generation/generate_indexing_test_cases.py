from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.path import Path as MplPath

from test_case_config import (
    BASE_GRID_CONFIG,
    MANDATORY_CASES,
    sample_random_case_definitions,
)


def get_project_root() -> Path:
    """
    Bestimmt den Projekt-Root.

    Datei liegt in:
    tests/indexing/test_data_generation/generate_indexing_test_cases.py
    """
    return Path(__file__).resolve().parents[3]


def get_output_dirs() -> tuple[Path, Path]:
    """
    Liefert die Zielordner für Testfälle und Testplots.
    """
    project_root = get_project_root()
    cases_dir = project_root / "tests" / "indexing" / "test_cases"
    results_dir = project_root / "tests" / "indexing" / "test_results"

    cases_dir.mkdir(parents=True, exist_ok=True)
    results_dir.mkdir(parents=True, exist_ok=True)

    return cases_dir, results_dir


def get_centered_even_doe_axis_indices(n: int) -> np.ndarray:
    """
    Erzeugt zentrierte DOE-Indizes für eine gerade Anzahl n.

    Beispiel:
        n = 10 -> [-5, -4, -3, -2, -1, 1, 2, 3, 4, 5]
        n = 50 -> [-25, ..., -1, 1, ..., 25]
    """
    if n <= 0 or n % 2 != 0:
        raise ValueError("n muss eine positive gerade Zahl sein.")

    half = n // 2
    return np.array(list(range(-half, 0)) + list(range(1, half + 1)), dtype=int)


def index_to_coordinate(idx: int, half_step: float) -> float:
    """
    Wandelt einen DOE-Index in eine ideale 1D-Koordinate um.

    Für ein nxm+1-Muster mit zusätzlichem Mittelpunkt gilt:
        idx = ±1 -> ±1 * half_step
        idx = ±2 -> ±3 * half_step
        idx = ±3 -> ±5 * half_step
        ...

    Also allgemein:
        coord = sign(idx) * (2*abs(idx) - 1) * half_step
    """
    if idx == 0:
        return 0.0

    return float(np.sign(idx) * (2 * abs(idx) - 1) * half_step)


def generate_base_grid(nx: int = 100, ny: int = 100, half_step: float = 20.0) -> np.ndarray:
    """
    Erzeugt ein großes ideales DOE-Raster.

    Rückgabeformat:
        Array der Form (N, 4):
        [idx_x, idx_y, x, y]
    """
    idx_x_vals = get_centered_even_doe_axis_indices(nx)
    idx_y_vals = get_centered_even_doe_axis_indices(ny)

    rows = []

    for idx_x in idx_x_vals:
        x = index_to_coordinate(idx_x, half_step)
        for idx_y in idx_y_vals:
            y = index_to_coordinate(idx_y, half_step)
            rows.append([idx_x, idx_y, x, y])

    # zusätzlicher Mittelpunkt
    rows.append([0, 0, 0.0, 0.0])

    return np.array(rows, dtype=float)


def mask_square(points_xy: np.ndarray, half_width: float, half_height: float) -> np.ndarray:
    x = points_xy[:, 0]
    y = points_xy[:, 1]
    return (np.abs(x) <= half_width) & (np.abs(y) <= half_height)


def mask_circle(points_xy: np.ndarray, radius: float) -> np.ndarray:
    x = points_xy[:, 0]
    y = points_xy[:, 1]
    return (x**2 + y**2) <= radius**2


def mask_diamond(points_xy: np.ndarray, radius_x: float, radius_y: float) -> np.ndarray:
    x = points_xy[:, 0]
    y = points_xy[:, 1]
    return (np.abs(x) / radius_x + np.abs(y) / radius_y) <= 1.0


def regular_polygon_vertices(
    n_sides: int,
    radius: float,
    rotation_deg: float = 0.0
) -> np.ndarray:
    """
    Erzeugt die Eckpunkte eines regelmäßigen Vielecks.
    """
    angles = np.linspace(0, 2 * np.pi, n_sides, endpoint=False) + np.deg2rad(rotation_deg)
    vertices = np.column_stack((radius * np.cos(angles), radius * np.sin(angles)))
    return vertices


def mask_polygon(points_xy: np.ndarray, vertices: np.ndarray) -> np.ndarray:
    polygon = MplPath(vertices)
    return polygon.contains_points(points_xy)


def apply_global_scaling(points_xy: np.ndarray, scale_x: float, scale_y: float) -> np.ndarray:
    transformed = points_xy.copy()
    transformed[:, 0] *= scale_x
    transformed[:, 1] *= scale_y
    return transformed


def apply_mild_perspective_like_distortion(
    points_xy: np.ndarray,
    strength_x: float = 0.0,
    strength_y: float = 0.0
) -> np.ndarray:
    """
    Milde achsenweise Verzerrung:
    - x-Skalierung hängt leicht von y ab
    - y-Skalierung hängt leicht von x ab

    Das simuliert eine schwache Asymmetrie / perspektivische Verzerrung.
    """
    transformed = points_xy.copy()

    x = transformed[:, 0]
    y = transformed[:, 1]

    max_abs_x = max(np.max(np.abs(x)), 1e-9)
    max_abs_y = max(np.max(np.abs(y)), 1e-9)

    transformed[:, 0] = x * (1.0 + strength_x * (y / max_abs_y))
    transformed[:, 1] = y * (1.0 + strength_y * (x / max_abs_x))

    return transformed


def apply_jitter(
    points_xy: np.ndarray,
    rng: np.random.Generator,
    full_step: float,
    jitter_fraction: float
) -> np.ndarray:
    """
    Verschiebt jeden Punkt leicht zufällig.

    jitter_fraction bezieht sich auf die volle Schrittweite.
    """
    transformed = points_xy.copy()
    max_jitter = jitter_fraction * full_step

    jitter = rng.uniform(-max_jitter, max_jitter, size=transformed.shape)
    transformed += jitter

    return transformed


def drop_random_points(
    points: np.ndarray,
    rng: np.random.Generator,
    missing_fraction: float,
    protect_center: bool = True
) -> tuple[np.ndarray, np.ndarray]:
    """
    Entfernt zufällig einen Anteil von Punkten.

    Rückgabe:
        remaining_points, dropped_points
    """
    if missing_fraction <= 0.0:
        return points.copy(), np.empty((0, points.shape[1]), dtype=points.dtype)

    points = points.copy()

    if protect_center:
        center_mask = (points[:, 0] == 0) & (points[:, 1] == 0)
    else:
        center_mask = np.zeros(len(points), dtype=bool)

    candidates = np.where(~center_mask)[0]
    n_drop = int(np.floor(missing_fraction * len(candidates)))

    if n_drop == 0:
        return points, np.empty((0, points.shape[1]), dtype=points.dtype)

    drop_indices = rng.choice(candidates, size=n_drop, replace=False)
    keep_mask = np.ones(len(points), dtype=bool)
    keep_mask[drop_indices] = False

    remaining = points[keep_mask]
    dropped = points[~keep_mask]

    return remaining, dropped


def ensure_center_present(points: np.ndarray) -> None:
    center_mask = (points[:, 0] == 0) & (points[:, 1] == 0)
    if not np.any(center_mask):
        raise ValueError("Der Mittelpunkt (0,0) muss erhalten bleiben.")


def plot_case(points: np.ndarray, title: str, save_path: Path) -> None:
    """
    Scatterplot mit Doppelindizes.
    Erwartetes Format points:
        [idx_x, idx_y, x, y]
    """
    fig, ax = plt.subplots(figsize=(10, 10))

    idx_x = points[:, 0].astype(int)
    idx_y = points[:, 1].astype(int)
    x = points[:, 2]
    y = points[:, 3]

    # Färbung nach Abstand vom Mittelpunkt nur zur Orientierung
    radius = np.sqrt(x**2 + y**2)
    scatter = ax.scatter(x, y, c=radius, cmap="viridis", s=30)
    fig.colorbar(scatter, ax=ax, label="distance from center")

    for ix, iy, px, py in zip(idx_x, idx_y, x, y):
        ax.text(px, py, f"({ix},{iy})", fontsize=7, ha="left", va="bottom")

    ax.set_title(title)
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_aspect("equal")

    # automatische Achsenskalierung mit kleinem Rand
    x_min, x_max = np.min(x), np.max(x)
    y_min, y_max = np.min(y), np.max(y)

    x_span = max(x_max - x_min, 1e-6)
    y_span = max(y_max - y_min, 1e-6)

    x_margin = 0.05 * x_span
    y_margin = 0.05 * y_span

    ax.set_xlim(x_min - x_margin, x_max + x_margin)
    ax.set_ylim(y_min - y_margin, y_max + y_margin)

    plt.tight_layout()
    save_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(save_path, dpi=200)
    plt.close(fig)


def save_case(points: np.ndarray, metadata: dict, case_name: str, cases_dir: Path, results_dir: Path) -> None:
    """
    Speichert:
    - .npy mit [idx_x, idx_y, x, y]
    - .json mit Metadaten
    - .png Plot
    """
    case_dir = cases_dir / case_name
    result_dir = results_dir / case_name
    case_dir.mkdir(parents=True, exist_ok=True)
    result_dir.mkdir(parents=True, exist_ok=True)

    npy_path = case_dir / f"{case_name}.npy"
    json_path = case_dir / f"{case_name}.json"
    plot_path = result_dir / f"{case_name}.png"

    np.save(npy_path, points)

    metadata_to_save = metadata.copy()
    metadata_to_save["array_format"] = ["idx_x", "idx_y", "x", "y"]
    metadata_to_save["num_points"] = int(len(points))

    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(metadata_to_save, f, indent=2)

    plot_case(points, title=case_name, save_path=plot_path)


def build_case(
    base_grid: np.ndarray,
    case_name: str,
    crop_kind: str,
    crop_params: dict,
    scale_x: float,
    scale_y: float,
    distortion_x: float,
    distortion_y: float,
    jitter_fraction: float,
    missing_fraction: float,
    seed: int,
    full_step: float
) -> tuple[np.ndarray, dict]:
    """
    Erzeugt einen einzelnen Testfall.
    """
    rng = np.random.default_rng(seed)

    points = base_grid.copy()
    idx = points[:, :2].copy()
    xy = points[:, 2:4].copy()

    # 1) Ausschneiden
    if crop_kind == "square":
        mask = mask_square(xy, **crop_params)
    elif crop_kind == "circle":
        mask = mask_circle(xy, **crop_params)
    elif crop_kind == "diamond":
        mask = mask_diamond(xy, **crop_params)
    elif crop_kind == "polygon":
        vertices = regular_polygon_vertices(**crop_params)
        mask = mask_polygon(xy, vertices)
    else:
        raise ValueError(f"Unbekannter crop_kind: {crop_kind}")

    # Mittelpunkt immer behalten
    center_mask = (idx[:, 0] == 0) & (idx[:, 1] == 0)
    mask = mask | center_mask

    idx = idx[mask]
    xy = xy[mask]

    # 2) globale Skalierung
    xy = apply_global_scaling(xy, scale_x=scale_x, scale_y=scale_y)

    # 3) milde Verzerrung
    xy = apply_mild_perspective_like_distortion(
        xy,
        strength_x=distortion_x,
        strength_y=distortion_y
    )

    # 4) zufällige Verschiebung
    xy = apply_jitter(
        xy,
        rng=rng,
        full_step=full_step,
        jitter_fraction=jitter_fraction
    )

    points_varied = np.column_stack((idx, xy))

    # 5) zufällige fehlende Punkte
    points_varied, dropped_points = drop_random_points(
        points_varied,
        rng=rng,
        missing_fraction=missing_fraction,
        protect_center=True
    )

    ensure_center_present(points_varied)

    metadata = {
        "case_name": case_name,
        "seed": seed,
        "base_grid": {
            "nx": int(np.max(np.abs(base_grid[:, 0])) * 2),
            "ny": int(np.max(np.abs(base_grid[:, 1])) * 2),
            "half_step": full_step / 2.0,
            "full_step": full_step
        },
        "transformations": {
            "crop_kind": crop_kind,
            "crop_params": crop_params,
            "scale_x": scale_x,
            "scale_y": scale_y,
            "distortion_x": distortion_x,
            "distortion_y": distortion_y,
            "jitter_fraction_of_full_step": jitter_fraction,
            "missing_fraction": missing_fraction
        },
        "dropped_indices": dropped_points[:, :2].astype(int).tolist() if len(dropped_points) > 0 else [],
        "notes": (
            "Index convention follows DOE centered signed indexing. "
            "The center point (0,0) is always retained."
        )
    }

    return points_varied, metadata


def main():
    cases_dir, results_dir = get_output_dirs()

    base_nx = BASE_GRID_CONFIG["nx"]
    base_ny = BASE_GRID_CONFIG["ny"]
    half_step = BASE_GRID_CONFIG["half_step"]
    full_step = 2.0 * half_step

    base_grid = generate_base_grid(
        nx=base_nx,
        ny=base_ny,
        half_step=half_step
    )

    random_cases = sample_random_case_definitions()
    all_case_definitions = [*MANDATORY_CASES, *random_cases]

    print("🔧 Generiere Indexing-Testfälle")
    print(f"Projekt-Root: {get_project_root()}")
    print(f"Testfälle:     {cases_dir}")
    print(f"Plots:         {results_dir}")
    print(f"Pflichtfälle:  {len(MANDATORY_CASES)}")
    print(f"Zufallsfälle:  {len(random_cases)}")
    print(f"Gesamtzahl:    {len(all_case_definitions)}")

    for cfg in all_case_definitions:
        case_name = cfg["case_name"]

        points_varied, metadata = build_case(
            base_grid=base_grid,
            case_name=case_name,
            crop_kind=cfg["crop_kind"],
            crop_params=cfg["crop_params"],
            scale_x=cfg["scale_x"],
            scale_y=cfg["scale_y"],
            distortion_x=cfg["distortion_x"],
            distortion_y=cfg["distortion_y"],
            jitter_fraction=cfg["jitter_fraction"],
            missing_fraction=cfg["missing_fraction"],
            seed=cfg["seed"],
            full_step=full_step
        )

        save_case(
            points=points_varied,
            metadata=metadata,
            case_name=case_name,
            cases_dir=cases_dir,
            results_dir=results_dir
        )

        print(
            f"  ✅ {case_name}: "
            f"{len(points_varied)} Punkte gespeichert | "
            f"{cfg['crop_kind']}, "
            f"jitter={cfg['jitter_fraction']:.3f}, "
            f"missing={cfg['missing_fraction']:.3f}"
        )

    print("🏁 Fertig.")


if __name__ == "__main__":
    main()
