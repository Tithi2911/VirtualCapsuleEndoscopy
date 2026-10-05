"""Frame quality assessment.

Missed lesions are often in parts of the bowel that were never properly seen
(motion blur, glare, fluid, the scope pressed against the wall). Scoring each
frame's quality lets the report show *where the review was blind*, not only
what was detected.
"""

from __future__ import annotations

import cv2
import numpy as np

from .config import AnalysisConfig
from .models import BlindSegment, Frame, QualityResult


def assess(image: np.ndarray, cfg: AnalysisConfig) -> QualityResult:
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    sharpness = float(cv2.Laplacian(gray, cv2.CV_64F).var())
    over = float(np.mean(gray >= 245))
    under = float(np.mean(gray <= 20))

    reasons = []
    if sharpness < cfg.min_sharpness:
        reasons.append("blurred")
    if over > cfg.max_overexposed_fraction:
        reasons.append("overexposed")
    if under > cfg.max_underexposed_fraction:
        reasons.append("too dark")
    return QualityResult(sharpness, over, under, informative=not reasons, reasons=reasons)


def blind_segments(
    frames: list[Frame], quality: list[QualityResult], cfg: AnalysisConfig
) -> list[BlindSegment]:
    """Contiguous stretches of non-informative frames lasting at least blind_segment_min_s."""
    segments: list[BlindSegment] = []
    start = None
    reasons: set[str] = set()
    for frame, q in zip(frames, quality):
        if not q.informative:
            if start is None:
                start = frame.timestamp_s
                reasons = set()
            reasons.update(q.reasons)
            end = frame.timestamp_s
        elif start is not None:
            _close(segments, start, end, reasons, cfg)
            start = None
    if start is not None:
        _close(segments, start, end, reasons, cfg)
    return segments


def _close(segments, start, end, reasons, cfg):
    if end - start >= cfg.blind_segment_min_s:
        segments.append(BlindSegment(start, end, ", ".join(sorted(reasons))))
