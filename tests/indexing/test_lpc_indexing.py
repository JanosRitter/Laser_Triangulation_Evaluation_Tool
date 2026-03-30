from __future__ import annotations

import csv
import json
from pathlib import Path
from datetime import datetime

import numpy as np
import pytest

from src.utils.lpc_indexing import assign_doe_indices, check_unique_indices


# ============================================================
# PFADHILFEN
# ============================================================
def get_project_root() -> Path:
    """
    Bestimmt den Projekt-Root.

    Datei liegt in:
    tests/indexing/test_lpc_indexing.py
    """
    return Path(__file__).resolve().parents[2]


def get_test_cases_dir() -> Path:
    return get_project_root() / "tests" / "indexing" / "test_cases"


def get_test_results_dir() -> Path:
    result_dir = get_project_root() / "tests" / "indexing" / "test_results"
    result_dir.mkdir(parents=True, exist_ok=True)
    return result_dir


# ============================================================
# LADEHILFEN
# ============================================================
def discover_case_files(test_cases_dir: Path) -> list[Path]:
    """
    Findet alle .npy-Testfälle rekursiv unter tests/indexing/test_cases.
    """
    case_files = sorted(test_cases_dir.rglob("*.npy"))
    return case_files


def load_case(case_path: Path) -> np.ndarray:
    """
    Lädt einen einzelnen Testfall.
    Erwartetes Format:
        [idx_x, idx_y, x, y]
    """
    arr = np.load(case_path)

    if arr.ndim != 2 or arr.shape[1] != 4:
        raise ValueError(
            f"Ungültiges Format in {case_path}. "
            f"Erwartet: (N, 4) mit [idx_x, idx_y, x, y], erhalten: {arr.shape}"
        )

    return arr.astype(float)


# ============================================================
# MATCHING / VERGLEICH
# ============================================================
def rounded_coord_key(x: float, y: float, decimals: int = 6) -> tuple[float, float]:
    """
    Erzeugt einen stabilen Koordinatenschlüssel mit Rundung.
    """
    return (round(float(x), decimals), round(float(y), decimals))


def build_index_lookup(points: np.ndarray, decimals: int = 6) -> dict[tuple[float, float], tuple[int, int]]:
    """
    Baut ein Lookup:
        (x, y) -> (idx_x, idx_y)

    Erwartetes Format:
        [idx_x, idx_y, x, y]
    """
    lookup = {}

    for row in points:
        idx_x = int(row[0])
        idx_y = int(row[1])
        x = float(row[2])
        y = float(row[3])

        key = rounded_coord_key(x, y, decimals=decimals)

        if key in lookup:
            raise ValueError(
                f"Doppelte Koordinate im Lookup gefunden: {key}. "
                f"Rundung evtl. zu grob oder Daten inkonsistent."
            )

        lookup[key] = (idx_x, idx_y)

    return lookup


def evaluate_case(case_array: np.ndarray, coord_decimals: int = 6) -> dict:
    """
    Führt die eigentliche Auswertung für einen Testfall durch.

    case_array Format:
        [idx_x, idx_y, x, y]
    """
    coords = case_array[:, 2:4].astype(float)

    predicted = assign_doe_indices(coords)

    if predicted.ndim != 2 or predicted.shape[1] != 4:
        raise ValueError(
            f"assign_doe_indices() lieferte unerwartetes Format: {predicted.shape}"
        )

    pred_indices = predicted[:, :2].astype(int)

    # Eindeutigkeit prüfen
    unique_predicted = check_unique_indices(pred_indices)

    pred_lookup = build_index_lookup(predicted, decimals=coord_decimals)

    # Vergleich: für jeden Ground-Truth-Punkt den vorhergesagten Index holen
    wrong_points = []
    missing_in_prediction = []
    correct_count = 0

    for row in case_array:
        true_idx_x = int(row[0])
        true_idx_y = int(row[1])
        x = float(row[2])
        y = float(row[3])

        key = rounded_coord_key(x, y, decimals=coord_decimals)

        if key not in pred_lookup:
            missing_in_prediction.append({
                "coord": [x, y],
                "true_index": [true_idx_x, true_idx_y]
            })
            continue

        pred_idx_x, pred_idx_y = pred_lookup[key]

        if pred_idx_x == true_idx_x and pred_idx_y == true_idx_y:
            correct_count += 1
        else:
            wrong_points.append({
                "coord": [x, y],
                "true_index": [true_idx_x, true_idx_y],
                "pred_index": [pred_idx_x, pred_idx_y]
            })

    total_points = len(case_array)
    wrong_count = len(wrong_points)
    missing_count = len(missing_in_prediction)

    accuracy = correct_count / total_points if total_points > 0 else 0.0
    passed = (wrong_count == 0) and (missing_count == 0) and unique_predicted

    return {
        "num_points": int(total_points),
        "num_correct": int(correct_count),
        "num_wrong": int(wrong_count),
        "num_missing_in_prediction": int(missing_count),
        "accuracy": float(accuracy),
        "predicted_indices_unique": bool(unique_predicted),
        "passed": bool(passed),
        "wrong_points": wrong_points,
        "missing_in_prediction": missing_in_prediction,
    }


# ============================================================
# REPORTING
# ============================================================
def print_case_report(case_name: str, result: dict, max_examples: int = 10) -> None:
    """
    Konsolenausgabe für einen Testfall.
    """
    print(f"\n📦 Testfall: {case_name}")
    print(f"  Punkte gesamt:         {result['num_points']}")
    print(f"  Korrekt indiziert:     {result['num_correct']}")
    print(f"  Falsch indiziert:      {result['num_wrong']}")
    print(f"  Nicht zugeordnet:      {result['num_missing_in_prediction']}")
    print(f"  Eindeutige Indizes:    {result['predicted_indices_unique']}")
    print(f"  Accuracy:              {100 * result['accuracy']:.2f}%")
    print(f"  Ergebnis:              {'✅ BESTANDEN' if result['passed'] else '❌ FEHLER'}")

    if result["wrong_points"]:
        print("  Beispiele falscher Zuordnungen:")
        for item in result["wrong_points"][:max_examples]:
            x, y = item["coord"]
            tix, tiy = item["true_index"]
            pix, piy = item["pred_index"]
            print(
                f"    coord=({x:.3f}, {y:.3f}) | "
                f"true=({tix},{tiy}) | pred=({pix},{piy})"
            )

    if result["missing_in_prediction"]:
        print("  Beispiele nicht zugeordneter Punkte:")
        for item in result["missing_in_prediction"][:max_examples]:
            x, y = item["coord"]
            tix, tiy = item["true_index"]
            print(
                f"    coord=({x:.3f}, {y:.3f}) | "
                f"true=({tix},{tiy})"
            )


def save_json_report(report: dict, results_dir: Path) -> Path:
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    report_path = results_dir / f"lpc_indexing_test_report_{timestamp}.json"

    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)

    return report_path


def save_csv_summary(report: dict, results_dir: Path) -> Path:
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    csv_path = results_dir / f"lpc_indexing_test_summary_{timestamp}.csv"

    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow([
            "case_name",
            "num_points",
            "num_correct",
            "num_wrong",
            "num_missing_in_prediction",
            "accuracy",
            "predicted_indices_unique",
            "passed",
        ])

        for case_name, result in report["cases"].items():
            writer.writerow([
                case_name,
                result["num_points"],
                result["num_correct"],
                result["num_wrong"],
                result["num_missing_in_prediction"],
                result["accuracy"],
                result["predicted_indices_unique"],
                result["passed"],
            ])

    return csv_path


# ============================================================
# PYTEST-INTEGRATION
# ============================================================
CASE_DIR = get_test_cases_dir()
CASE_FILES = discover_case_files(CASE_DIR)


def test_indexing_cases_exist():
    assert len(CASE_FILES) > 0, f"Keine .npy-Testfälle gefunden in: {CASE_DIR}"


@pytest.mark.parametrize("case_path", CASE_FILES, ids=[p.stem for p in CASE_FILES])
def test_lpc_indexing_case(case_path: Path):
    case_array = load_case(case_path)
    result = evaluate_case(case_array, coord_decimals=6)

    assert result["predicted_indices_unique"], (
        f"{case_path.stem}: Doppelte vorhergesagte Indizes gefunden."
    )

    assert result["num_missing_in_prediction"] == 0, (
        f"{case_path.stem}: {result['num_missing_in_prediction']} Punkte "
        f"konnten nicht zugeordnet werden."
    )

    assert result["num_wrong"] == 0, (
        f"{case_path.stem}: {result['num_wrong']} Punkte falsch indiziert. "
        f"Beispiele: {result['wrong_points'][:5]}"
    )

    assert result["passed"], f"{case_path.stem}: Test nicht bestanden."


# ============================================================
# BATCH-RUNNER (optional weiter nutzbar)
# ============================================================
def main():
    project_root = get_project_root()
    test_cases_dir = get_test_cases_dir()
    test_results_dir = get_test_results_dir()

    print("🔧 LPC Indexing Batch-Test gestartet")
    print(f"  Projekt-Root:   {project_root}")
    print(f"  Testfälle:      {test_cases_dir}")
    print(f"  Ergebnisse:     {test_results_dir}")

    case_files = discover_case_files(test_cases_dir)

    if not case_files:
        raise FileNotFoundError(
            f"Keine .npy-Testfälle gefunden in: {test_cases_dir}"
        )

    print(f"  Gefundene Fälle: {len(case_files)}")

    report = {
        "project_root": str(project_root),
        "test_cases_dir": str(test_cases_dir),
        "test_results_dir": str(test_results_dir),
        "num_cases": len(case_files),
        "cases": {},
        "summary": {}
    }

    passed_cases = 0
    failed_cases = 0

    for case_path in case_files:
        case_name = case_path.stem

        try:
            case_array = load_case(case_path)
            result = evaluate_case(case_array, coord_decimals=6)

            report["cases"][case_name] = {
                "case_path": str(case_path),
                **result
            }

            print_case_report(case_name, result)

            if result["passed"]:
                passed_cases += 1
            else:
                failed_cases += 1

        except Exception as exc:
            failed_cases += 1
            report["cases"][case_name] = {
                "case_path": str(case_path),
                "passed": False,
                "error": str(exc)
            }

            print(f"\n📦 Testfall: {case_name}")
            print("  Ergebnis: ❌ FEHLER BEIM TESTLAUF")
            print(f"  Fehler:   {exc}")

    report["summary"] = {
        "passed_cases": passed_cases,
        "failed_cases": failed_cases,
        "all_passed": failed_cases == 0
    }

    json_report_path = save_json_report(report, test_results_dir)
    csv_summary_path = save_csv_summary(report, test_results_dir)

    print("\n🏁 Gesamtergebnis")
    print(f"  Bestandene Fälle: {passed_cases}")
    print(f"  Fehlgeschlagene:  {failed_cases}")
    print(f"  Alle bestanden:   {failed_cases == 0}")
    print(f"  JSON-Report:      {json_report_path}")
    print(f"  CSV-Zusammenf.:   {csv_summary_path}")


if __name__ == "__main__":
    main()