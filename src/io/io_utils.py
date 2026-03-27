from pathlib import Path
import numpy as np
from PIL import Image

from src.utils.path_utils import (
    IMAGE_INPUT_DIR,
    ensure_directory,
    get_output_folder_for_input,
)


def resolve_input_folder(folder_name_or_path: str | Path) -> Path:
    """
    Löst einen Eingabepfad auf.

    Regeln:
    - Absoluter Pfad bleibt absolut
    - Relativer Pfad wird relativ zu data/input/images interpretiert
    """
    folder = Path(folder_name_or_path)

    if folder.is_absolute():
        resolved = folder
    else:
        resolved = IMAGE_INPUT_DIR / folder

    if not resolved.exists():
        raise FileNotFoundError(f"Eingabeordner nicht gefunden: {resolved}")

    if not resolved.is_dir():
        raise NotADirectoryError(f"Pfad ist kein Ordner: {resolved}")

    return resolved


def png_to_uint8_array(png_path: Path) -> np.ndarray:
    """
    Liest eine PNG-Datei ein und konvertiert sie zu einem 2D-uint8-Grauwertbild.
    """
    with Image.open(png_path) as img:
        gray = img.convert("L")
        array = np.array(gray, dtype=np.uint8)

    return array


def load_or_create_npy_for_png(png_path: Path, overwrite: bool = False) -> np.ndarray:
    """
    Lädt die .npy-Datei zu einer PNG, falls vorhanden.
    Andernfalls wird sie aus der PNG erzeugt und gespeichert.
    """
    npy_path = png_path.with_suffix(".npy")

    if npy_path.exists() and not overwrite:
        print(f"Vorhandene .npy geladen: {npy_path.name}")
        return np.load(npy_path)

    brightness_array = png_to_uint8_array(png_path)
    np.save(npy_path, brightness_array)
    print(f"Neue .npy gespeichert: {npy_path.name}")

    return brightness_array


def load_or_create_npy_folder(folder_name_or_path: str | Path, overwrite: bool = False):
    """
    Verarbeitet alle PNG-Dateien in einem Ordner.

    Returns
    -------
    arrays : dict[str, np.ndarray]
        Geladene Arrays
    folder_path : Path
        Aufgelöster Input-Ordner
    """
    folder_path = resolve_input_folder(folder_name_or_path)

    png_files = sorted(folder_path.glob("*.png"))
    if not png_files:
        raise FileNotFoundError(f"Keine PNG-Dateien gefunden in: {folder_path}")

    arrays = {}

    for png_path in png_files:
        array = load_or_create_npy_for_png(png_path, overwrite=overwrite)
        arrays[png_path.stem] = array

    return arrays, folder_path


def save_npy_array(
    array: np.ndarray,
    folder_path: str | Path,
    file_name: str,
    overwrite: bool = False
) -> Path:
    """
    Speichert ein Array als .npy in einen beliebigen Zielordner.
    """
    folder = ensure_directory(Path(folder_path))

    if not file_name.lower().endswith(".npy"):
        file_name = f"{file_name}.npy"

    file_path = folder / file_name

    if file_path.exists() and not overwrite:
        print(f"Datei existiert bereits, nicht überschrieben: {file_path.name}")
        return file_path

    np.save(file_path, array)
    print(f"Array gespeichert: {file_path.name}")
    return file_path


def save_result_for_input_folder(
    array: np.ndarray,
    input_folder: str | Path,
    file_name: str,
    overwrite: bool = False
) -> Path:
    """
    Speichert ein Verarbeitungsergebnis im Output-Ordner, der zum Input-Timestamp gehört.

    Beispiel:
    input/images/2026-03-24_14-57-19/...
    -> output/2026-03-24_14-57-19/<file_name>.npy
    """
    output_folder = get_output_folder_for_input(input_folder)
    return save_npy_array(array, output_folder, file_name, overwrite=overwrite)


def load_npy_file(folder_name_or_path: str | Path, file_stem: str) -> np.ndarray:
    """
    Lädt eine bestehende .npy-Datei aus einem Input-Ordner.
    """
    folder_path = resolve_input_folder(folder_name_or_path)
    npy_path = folder_path / f"{file_stem}.npy"

    if not npy_path.exists():
        raise FileNotFoundError(f".npy-Datei nicht gefunden: {npy_path}")

    return np.load(npy_path)