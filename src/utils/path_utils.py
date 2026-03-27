from pathlib import Path

# Projektwurzel automatisch bestimmen
PROJECT_ROOT = Path(__file__).resolve().parents[2]

# Basisordner
DATA_DIR = PROJECT_ROOT / "data"
INPUT_DIR = DATA_DIR / "input"
OUTPUT_DIR = DATA_DIR / "output"

# Input-Unterordner
IMAGE_INPUT_DIR = INPUT_DIR / "images"
CALIBRATION_INPUT_DIR = INPUT_DIR / "calibration"


def ensure_directory(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path


def get_output_folder_for_input(input_folder: str | Path) -> Path:
    """
    Erzeugt den passenden Output-Ordner zu einem Input-Ordner.

    Beispiele:
    - input/images/2026-03-24_14-57-19  -> output/2026-03-24_14-57-19
    - "2026-03-24_14-57-19"             -> output/2026-03-24_14-57-19
    """
    folder = Path(input_folder)
    timestamp_name = folder.name
    output_folder = OUTPUT_DIR / timestamp_name
    return ensure_directory(output_folder)