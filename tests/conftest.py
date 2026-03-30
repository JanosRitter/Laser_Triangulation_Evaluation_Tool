from pathlib import Path
import pytest


def get_project_root() -> Path:
    return Path(__file__).resolve().parents[1]


@pytest.fixture
def project_root() -> Path:
    return get_project_root()


@pytest.fixture
def indexing_test_cases_dir(project_root: Path) -> Path:
    return project_root / "tests" / "indexing" / "test_cases"


@pytest.fixture
def indexing_test_results_dir(project_root: Path) -> Path:
    result_dir = project_root / "tests" / "indexing" / "test_results"
    result_dir.mkdir(parents=True, exist_ok=True)
    return result_dir
