"""Basic sanity tests for the PathSnap project structure."""

from pathlib import Path

# Project root = the folder that contains the "tests" folder
ROOT = Path(__file__).resolve().parent.parent


def test_readme_exists():
    assert (ROOT / "README.md").is_file()


def test_config_file_exists_and_not_empty():
    config = ROOT / "config.yaml"
    assert config.is_file()
    assert config.stat().st_size > 0


def test_requirements_file_exists():
    assert (ROOT / "requirements.txt").is_file()


def test_main_folders_exist():
    for folder in ["app", "pathsnap", "preprocessing", "scripts"]:
        assert (ROOT / folder).is_dir(), f"Missing folder: {folder}"


def test_preprocessing_entry_script_exists():
    assert (ROOT / "run_preprocessing.py").is_file()
