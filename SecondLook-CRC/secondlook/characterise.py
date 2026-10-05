"""Lesion characterisation (CADx) plug-in point.

A Characteriser receives a finding plus the images of the frames it was seen
in, and returns a Characterisation. Using several frames, not just the best
one, makes the prediction more stable and lets the classifier report how much
the frames agree.

Without a trained classifier the NullCharacteriser is used and every lesion is
reported as uncharacterised. Optical diagnosis is the clinically riskiest output
of this software and must never be produced by an untrained rule. See
secondlook/diagnosis/ and docs/DIAGNOSIS.md.
"""

from __future__ import annotations

from typing import Protocol

import numpy as np

from .models import Characterisation, Finding


class Characteriser(Protocol):
    name: str
    version: str

    def characterise(self, finding: Finding, frames: dict[int, np.ndarray]) -> Characterisation:
        """`frames` maps original frame index -> BGR image for frames in which the finding was detected
        (at least the finding's best frame). Detection bboxes are in those images' pixel coordinates."""
        ...


class NullCharacteriser:
    name = "none"
    version = "-"

    def characterise(self, finding: Finding, frames: dict[int, np.ndarray]) -> Characterisation:
        return Characterisation()
