from src.config.config import EVALUATION_MODE, INPUT_FOLDER
from src.pipelines.doe_pipeline import run_doe_folder


def main():
    print("🔧 Evaluation gestartet")

    if EVALUATION_MODE == "doe":
        run_doe_folder(INPUT_FOLDER)
    else:
        raise ValueError(f"Unbekannter EVALUATION_MODE: {EVALUATION_MODE}")


if __name__ == "__main__":
    main()