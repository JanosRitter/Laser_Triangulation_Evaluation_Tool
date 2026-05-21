from pathlib import Path

# Projektwurzel automatisch bestimmen
PROJECT_ROOT = Path(__file__).resolve().parents[2]

# Neue Datenstruktur:
#
# data/
#   MESSREIHE/
#     images/
#     crops/
#     laser_calibration/
#     results/
#     run_metadata.json
#     frame_table.csv
DATA_DIR = PROJECT_ROOT / "data"
RUNS_DIR = DATA_DIR


def ensure_directory(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path


def resolve_run_folder(folder_name_or_path: str | Path) -> Path:
    """
    Löst einen Messreihen-/Run-Ordner auf.

    Regeln:
    - Absoluter Pfad bleibt absolut.
    - Relativer Pfad wird relativ zu data/ interpretiert.

    Beispiel:
        "20260520_100303_robot_measurement"
        -> data/20260520_100303_robot_measurement
    """
    folder = Path(folder_name_or_path)

    if folder.is_absolute():
        resolved = folder
    else:
        resolved = RUNS_DIR / folder

    if not resolved.exists():
        raise FileNotFoundError(f"Run-Ordner nicht gefunden: {resolved}")

    if not resolved.is_dir():
        raise NotADirectoryError(f"Pfad ist kein Ordner: {resolved}")

    return resolved


def get_results_folder_for_run(run_folder: str | Path) -> Path:
    """
    Gibt den Results-Ordner einer Messreihe zurück und legt ihn an.

    Beispiel:
        data/20260520_100303_robot_measurement
        -> data/20260520_100303_robot_measurement/results
    """
    run_folder = resolve_run_folder(run_folder)
    return ensure_directory(run_folder / "results")


# -------------------------------------------------------------------------
# Legacy-Kompatibilität
# -------------------------------------------------------------------------

def get_output_folder_for_input(input_folder: str | Path) -> Path:
    """
    Legacy-Name für bestehenden Pipeline-Code.

    Früher:
        data/input/images/MESSREIHE
        -> data/output/MESSREIHE

    Jetzt:
        data/MESSREIHE
        -> data/MESSREIHE/results
    """
    return get_results_folder_for_run(input_folder)


# Alte Konstanten nicht mehr aktiv verwenden.
# Sie bleiben nur stehen, damit alte Imports nicht sofort brechen.
INPUT_DIR = DATA_DIR
OUTPUT_DIR = DATA_DIR
IMAGE_INPUT_DIR = DATA_DIR
CALIBRATION_INPUT_DIR = DATA_DIR