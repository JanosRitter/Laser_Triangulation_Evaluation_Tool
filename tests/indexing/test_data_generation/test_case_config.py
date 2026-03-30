from __future__ import annotations

from itertools import product
import numpy as np


# ============================================================
# BASIS-GRID
# ============================================================
BASE_GRID_CONFIG = {
    "nx": 100,
    "ny": 100,
    "half_step": 20.0,
}


# ============================================================
# PFLICHTFÄLLE
# ============================================================
MANDATORY_CASES = [
    {
        "case_name": "indexing_test_1",
        "crop_kind": "square",
        "crop_params": {"half_width": 140.0, "half_height": 140.0},
        "scale_x": 1.00,
        "scale_y": 1.00,
        "distortion_x": 0.00,
        "distortion_y": 0.00,
        "jitter_fraction": 0.015,
        "missing_fraction": 0.00,
        "seed": 101
    },
    {
        "case_name": "indexing_test_2",
        "crop_kind": "square",
        "crop_params": {"half_width": 180.0, "half_height": 140.0},
        "scale_x": 1.03,
        "scale_y": 0.98,
        "distortion_x": 0.02,
        "distortion_y": 0.00,
        "jitter_fraction": 0.020,
        "missing_fraction": 0.00,
        "seed": 102
    },
    {
        "case_name": "indexing_test_3",
        "crop_kind": "circle",
        "crop_params": {"radius": 185.0},
        "scale_x": 1.00,
        "scale_y": 1.02,
        "distortion_x": 0.025,
        "distortion_y": -0.015,
        "jitter_fraction": 0.025,
        "missing_fraction": 0.003,
        "seed": 103
    },
    {
        "case_name": "indexing_test_4",
        "crop_kind": "diamond",
        "crop_params": {"radius_x": 220.0, "radius_y": 170.0},
        "scale_x": 1.05,
        "scale_y": 0.95,
        "distortion_x": 0.035,
        "distortion_y": 0.020,
        "jitter_fraction": 0.030,
        "missing_fraction": 0.006,
        "seed": 104
    },
    {
        "case_name": "indexing_test_5",
        "crop_kind": "polygon",
        "crop_params": {"n_sides": 6, "radius": 220.0, "rotation_deg": 10.0},
        "scale_x": 1.06,
        "scale_y": 0.94,
        "distortion_x": 0.040,
        "distortion_y": -0.025,
        "jitter_fraction": 0.035,
        "missing_fraction": 0.010,
        "seed": 105
    },
]


# ============================================================
# PARAMETERRAUM FÜR ZUSÄTZLICHE FÄLLE
# ============================================================
RANDOM_CASE_SPACE = {
    "crop_kind": ["square", "circle", "diamond", "polygon"],
    "scale_x": [1.00, 1.02, 1.10],
    "scale_y": [1.00, 0.98, 0.92, 1.03],
    "distortion_x": [0.00, 0.015, 0.04],
    "distortion_y": [-0.05, 0.00, 0.05],
    "jitter_fraction": [0.015, 0.025, 0.05],
    "missing_fraction": [0.00, 0.010, 0.020, 0.040],
}


# ============================================================
# PARAMETERBIBLIOTHEKEN FÜR CROP-SHAPES
# ============================================================
CROP_PARAM_LIBRARY = {
    "square": [
        {"half_width": 140.0, "half_height": 140.0},
        {"half_width": 180.0, "half_height": 140.0},
        {"half_width": 200.0, "half_height": 180.0},
    ],
    "circle": [
        {"radius": 170.0},
        {"radius": 190.0},
        {"radius": 220.0},
    ],
    "diamond": [
        {"radius_x": 180.0, "radius_y": 150.0},
        {"radius_x": 220.0, "radius_y": 170.0},
        {"radius_x": 260.0, "radius_y": 180.0},
    ],
    "polygon": [
        {"n_sides": 5, "radius": 190.0, "rotation_deg": 0.0},
        {"n_sides": 6, "radius": 220.0, "rotation_deg": 10.0},
        {"n_sides": 8, "radius": 240.0, "rotation_deg": 20.0},
    ],
}


# ============================================================
# GENERIERUNGSKONFIGURATION
# ============================================================
RANDOM_GENERATION_CONFIG = {
    "n_random_cases": 95,
    "random_seed": 12345,
    "case_name_start_index": len(MANDATORY_CASES) + 1,
}


def is_reasonable_case_definition(cfg: dict) -> bool:
    """
    Filtert offensichtlich unpassende Kombinationen aus.
    """
    # sehr harte Kombinationen erstmal vermeiden
    if cfg["jitter_fraction"] > 0.03 and cfg["missing_fraction"] > 0.006:
        return False

    if abs(cfg["distortion_x"]) > 0.025 and abs(cfg["distortion_y"]) > 0.015:
        return False

    if cfg["crop_kind"] == "square" and cfg["missing_fraction"] > 0.006:
        return False

    return True


def generate_candidate_case_definitions() -> list[dict]:
    """
    Erzeugt den vollständigen Kandidatenraum sinnvoller Fallkonfigurationen.
    """
    candidates = []

    for (
        crop_kind,
        scale_x,
        scale_y,
        distortion_x,
        distortion_y,
        jitter_fraction,
        missing_fraction,
    ) in product(
        RANDOM_CASE_SPACE["crop_kind"],
        RANDOM_CASE_SPACE["scale_x"],
        RANDOM_CASE_SPACE["scale_y"],
        RANDOM_CASE_SPACE["distortion_x"],
        RANDOM_CASE_SPACE["distortion_y"],
        RANDOM_CASE_SPACE["jitter_fraction"],
        RANDOM_CASE_SPACE["missing_fraction"],
    ):
        for crop_params in CROP_PARAM_LIBRARY[crop_kind]:
            cfg = {
                "crop_kind": crop_kind,
                "crop_params": crop_params,
                "scale_x": scale_x,
                "scale_y": scale_y,
                "distortion_x": distortion_x,
                "distortion_y": distortion_y,
                "jitter_fraction": jitter_fraction,
                "missing_fraction": missing_fraction,
            }

            if is_reasonable_case_definition(cfg):
                candidates.append(cfg)

    return candidates


def sample_random_case_definitions() -> list[dict]:
    """
    Wählt reproduzierbar eine begrenzte Anzahl zufälliger Testfälle aus.
    """
    rng = np.random.default_rng(RANDOM_GENERATION_CONFIG["random_seed"])
    candidates = generate_candidate_case_definitions()

    n_random_cases = min(
        RANDOM_GENERATION_CONFIG["n_random_cases"],
        len(candidates)
    )

    chosen_indices = rng.choice(len(candidates), size=n_random_cases, replace=False)

    sampled = []
    start_index = RANDOM_GENERATION_CONFIG["case_name_start_index"]

    for i, idx in enumerate(chosen_indices):
        cfg = candidates[idx].copy()
        case_number = start_index + i
        cfg["case_name"] = f"indexing_test_{case_number}"
        cfg["seed"] = int(RANDOM_GENERATION_CONFIG["random_seed"] + case_number)
        sampled.append(cfg)

    return sampled

