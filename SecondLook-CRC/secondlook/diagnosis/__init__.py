"""AI optical diagnosis: classify detected lesions as benign, precancerous or cancerous."""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

from .taxonomy import CATEGORIES, DISPLAY_NAME, DiagnosticCategory, category_for

if TYPE_CHECKING:
    from .classifier import OnnxLesionClassifier


def load_classifier(model_path: Path | str, **kwargs) -> "OnnxLesionClassifier":
    """Trained ONNX lesion classifier plus its JSON sidecar. Needs onnxruntime (the 'model' extra)."""
    from .classifier import OnnxLesionClassifier

    return OnnxLesionClassifier(model_path, **kwargs)


__all__ = ["CATEGORIES", "DISPLAY_NAME", "DiagnosticCategory", "category_for", "load_classifier"]
