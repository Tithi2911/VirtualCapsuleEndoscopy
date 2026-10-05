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

from typing import Optional, Protocol

import numpy as np

from .models import Characterisation, Finding


class Characteriser(Protocol):
    """Optional extras, read with getattr: `info` (dict of provenance shown in the report and
    result JSON) and `unsupported_reason(cfg)` (why it must not run on a recording analysed
    with `cfg`, e.g. a colonoscopy model on a capsule study, or None)."""

    name: str
    version: str

    def characterise(self, finding: Finding, frames: dict[int, np.ndarray]) -> Characterisation:
        """`frames` maps original frame index -> BGR image for frames in which the finding was detected
        (at least the finding's best frame), at the analysis working size. Detection bboxes are in
        those images' pixel coordinates."""
        ...


def unsupported_reason(characteriser: Characteriser, cfg) -> Optional[str]:
    check = getattr(characteriser, "unsupported_reason", None)
    return check(cfg) if callable(check) else None


def not_applied(characteriser: Characteriser, reason: str) -> Characterisation:
    """What a finding gets when the characteriser must not be used on this recording at all."""
    return Characterisation(
        model=characteriser.name,
        model_version=characteriser.version,
        abstained=True,
        abstain_reason=reason,
        note=f"No AI category given: {reason}. The lesion needs assessment by the reviewer.",
    )


class NullCharacteriser:
    name = "none"
    version = "-"

    def characterise(self, finding: Finding, frames: dict[int, np.ndarray]) -> Characterisation:
        return Characterisation()
