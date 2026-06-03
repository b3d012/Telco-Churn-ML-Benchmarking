"""Shared utility functions used across the modeling workflow."""

from __future__ import annotations

import re
from pathlib import Path
from typing import Iterable

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline


def ensure_directory(path: Path) -> Path:
    """Create a directory if it does not already exist."""
    path.mkdir(parents=True, exist_ok=True)
    return path


def ensure_parent_dir(path: Path) -> Path:
    """Create the parent directory for a file path."""
    ensure_directory(path.parent)
    return path


def save_artifact(obj, path: Path) -> Path:
    """Persist a Python object with joblib."""
    ensure_parent_dir(path)
    joblib.dump(obj, path)
    return path


def load_artifact(path: Path):
    """Load a joblib artifact from disk."""
    return joblib.load(path)


def flatten(iterables: Iterable[Iterable]):
    """Flatten one level of nesting."""
    for iterable in iterables:
        yield from iterable


def save_figure(fig: plt.Figure, path: Path, dpi: int = 300) -> Path:
    """Save a matplotlib figure and close it."""
    ensure_parent_dir(path)
    fig.tight_layout()
    fig.savefig(path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    return path


def normalize_text_values(value):
    """Normalize messy text by stripping, collapsing spaces, and lowercasing for matching."""
    if isinstance(value, pd.Series):
        return value.astype(str).str.replace(r"\s+", " ", regex=True).str.strip()
    if pd.isna(value):
        return value
    text = str(value).replace("\u00a0", " ")
    text = re.sub(r"\s+", " ", text).strip()
    return text


def normalize_key(value):
    """Return a lowercase normalized key for mapping."""
    if pd.isna(value):
        return None
    text = normalize_text_values(value)
    return str(text).lower() if text != "" else ""


def print_section_header(title: str) -> None:
    """Print a readable section header for CLI runs."""
    line = "=" * max(len(title), 12)
    print(f"\n{line}\n{title}\n{line}")


def get_feature_names_from_column_transformer(
    preprocessor: ColumnTransformer,
    input_features: list[str] | None = None,
) -> list[str]:
    """Safely extract feature names from a fitted ColumnTransformer."""
    try:
        if input_features is not None:
            feature_names = preprocessor.get_feature_names_out(input_features)
        else:
            feature_names = preprocessor.get_feature_names_out()
        return [str(name) for name in feature_names]
    except Exception:
        names: list[str] = []
        for name, transformer, columns in preprocessor.transformers_:
            if name == "remainder" and transformer == "drop":
                continue
            if transformer == "drop":
                continue
            if transformer == "passthrough":
                names.extend(list(columns))
                continue
            if isinstance(transformer, Pipeline):
                try:
                    pipeline_names = transformer.get_feature_names_out(columns)
                    names.extend([str(n) for n in pipeline_names])
                    continue
                except Exception:
                    pass
            if isinstance(columns, (list, tuple, np.ndarray, pd.Index)):
                names.extend([str(c) for c in columns])
            else:
                names.append(str(columns))
        return names

