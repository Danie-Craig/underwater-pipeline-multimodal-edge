"""Multimodal underwater pipeline-inspection pipeline (edge-optimized).

This package holds the reusable building blocks; the runnable entry points
live in ``scripts/``. A single small helper, :func:`load_config`, gives every
script one consistent way to read ``configs/model_config.yaml``.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

__all__ = ["REPO_ROOT", "load_config"]

# Repository root = parent of this ``src`` directory. Used to resolve the
# relative paths stored in the config (data/, models/, results/, ...).
REPO_ROOT: Path = Path(__file__).resolve().parents[1]


def load_config(path: str | Path = "configs/model_config.yaml") -> dict[str, Any]:
    """Load the project YAML config as a plain dict.

    Relative ``path`` values are resolved against the repository root, so the
    config loads correctly no matter which directory a script is invoked from.
    """
    import yaml  # local import: keeps ``import src`` light

    cfg_path = Path(path)
    if not cfg_path.is_absolute():
        cfg_path = REPO_ROOT / cfg_path
    with cfg_path.open("r", encoding="utf-8") as fh:
        return yaml.safe_load(fh)
