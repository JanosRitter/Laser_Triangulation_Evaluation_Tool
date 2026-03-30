import subprocess
import sys
from pathlib import Path


def run_pytest():
    project_root = Path(__file__).resolve().parent

    print("🔧 Starte pytest...\n")

    result = subprocess.run(
        [sys.executable, "-m", "pytest", "-v"],
        cwd=project_root
    )

    if result.returncode == 0:
        print("\n✅ Alle Tests bestanden")
    else:
        print("\n❌ Tests fehlgeschlagen")


if __name__ == "__main__":
    run_pytest()