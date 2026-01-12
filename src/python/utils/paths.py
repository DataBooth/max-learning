"""Path utilities for the MAX learning repository.

Provides helper functions for resolving project paths and managing sys.path.
"""

import sys
from pathlib import Path


def get_project_root() -> Path:
    """Get the root directory of the max-learning project.

    Searches upward from the current file for project marker files:
    - pixi.toml (primary marker)
    - .git directory (fallback)

    Returns:
        Path to the project root directory.

    Raises:
        RuntimeError: If project root cannot be found.
    """
    # Start from this file's directory
    current = Path(__file__).resolve().parent

    # Search upward for project markers
    for parent in [current] + list(current.parents):
        # Check for pixi.toml (primary marker)
        if (parent / "pixi.toml").exists():
            return parent
        # Check for .git directory (fallback)
        if (parent / ".git").exists():
            return parent

    # If we get here, we couldn't find the project root
    raise RuntimeError(
        "Could not find project root. "
        "Expected to find pixi.toml or .git directory in parent directories."
    )


def add_project_root_to_path() -> Path:
    """Add project root to sys.path if not already present.

    This enables importing from src/ and examples/ directories.

    Returns:
        Path to the project root directory.
    """
    project_root = get_project_root()
    project_root_str = str(project_root)

    if project_root_str not in sys.path:
        sys.path.insert(0, project_root_str)

    return project_root


def get_examples_dir() -> Path:
    """Get the examples directory.

    Returns:
        Path to examples/python/ directory.
    """
    return get_project_root() / "examples" / "python"


def get_tests_dir() -> Path:
    """Get the tests directory.

    Returns:
        Path to tests/python/ directory.
    """
    return get_project_root() / "tests" / "python"


def get_models_dir() -> Path:
    """Get the models directory.

    Returns:
        Path to models/ directory.
    """
    return get_project_root() / "models"
