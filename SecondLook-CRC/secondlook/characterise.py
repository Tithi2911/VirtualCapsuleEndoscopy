"""Lesion characterisation (CADx) plug-in point.

Estimating histology or malignancy risk from images is the clinically riskiest
output this product could make, so it is deliberately empty until a model has
been trained on histology-confirmed data and validated (see docs/DESIGN.md,
"Characterisation"). The interface is fixed now so that reports, audit and the
review UI already carry the fields.
"""

from __future__ import annotations

from typing import Protocol

import numpy as np

from .models import Characterisation, Finding


class Characteriser(Protocol):
    name: str

    def characterise(self, finding: Finding, best_image: np.ndarray) -> Characterisation: ...


class NullCharacteriser:
    name = "none"

    def characterise(self, finding: Finding, best_image: np.ndarray) -> Characterisation:
        return Characterisation()
