"""Core data types shared across the pipeline."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Optional

import numpy as np


class Modality(str, Enum):
    COLONOSCOPY = "colonoscopy"
    CAPSULE = "capsule"


class FindingStatus(str, Enum):
    # A clinician report was supplied and this finding matches a reported lesion.
    REPORTED = "reported"
    # A clinician report was supplied and nothing in it matches this finding.
    POTENTIALLY_MISSED = "potentially_missed"
    # No clinician report was supplied, so the finding cannot be compared.
    UNREVIEWED = "unreviewed"


@dataclass
class Frame:
    index: int  # position in the original recording (before sampling)
    timestamp_s: float
    image: np.ndarray  # BGR uint8
    source: str  # file name or "video"


@dataclass
class QualityResult:
    sharpness: float  # variance of Laplacian
    overexposed_fraction: float
    underexposed_fraction: float
    informative: bool
    reasons: list[str] = field(default_factory=list)


@dataclass
class Detection:
    frame_index: int
    timestamp_s: float
    bbox: tuple[int, int, int, int]  # x, y, w, h in pixels
    score: float  # 0..1, detector confidence
    label: str
    # Bool HxW region and float32 HxW 0..1 per-pixel evidence used for the explanation.
    # Released (set to None) for all but each finding's best detection to bound memory.
    mask: Optional[np.ndarray]
    heatmap: Optional[np.ndarray]
    evidence: dict[str, float] = field(default_factory=dict)  # named features behind the score


@dataclass
class Characterisation:
    """Optical-diagnosis output (CADx). Empty unless a validated model is configured."""

    model: Optional[str] = None
    histology_prediction: Optional[str] = None  # e.g. "adenomatous" / "hyperplastic"
    malignancy_risk: Optional[float] = None  # 0..1
    morphology: Optional[str] = None  # e.g. Paris classification
    note: str = "No validated characterisation model configured; lesion is uncharacterised."


@dataclass
class Finding:
    finding_id: str
    detections: list[Detection]
    status: FindingStatus = FindingStatus.UNREVIEWED
    matched_report_id: Optional[str] = None
    characterisation: Characterisation = field(default_factory=Characterisation)
    rationale: list[str] = field(default_factory=list)

    @property
    def first(self) -> Detection:
        return self.detections[0]

    @property
    def last(self) -> Detection:
        return self.detections[-1]

    @property
    def best(self) -> Detection:
        return max(self.detections, key=lambda d: d.score)

    @property
    def peak_score(self) -> float:
        return self.best.score

    @property
    def mean_score(self) -> float:
        return float(np.mean([d.score for d in self.detections]))

    @property
    def duration_s(self) -> float:
        return self.last.timestamp_s - self.first.timestamp_s


@dataclass
class BlindSegment:
    """A stretch of the recording with too few informative frames to review."""

    start_s: float
    end_s: float
    reason: str

    @property
    def duration_s(self) -> float:
        return self.end_s - self.start_s


@dataclass
class ReportedFinding:
    """A lesion documented by the endoscopist in the original procedure report."""

    report_id: str
    time_s: Optional[float] = None
    frame_index: Optional[int] = None
    location: Optional[str] = None
    note: Optional[str] = None


def to_jsonable(obj):
    """Dataclass -> dict, dropping numpy arrays (masks/heatmaps live in image files)."""
    if isinstance(obj, np.ndarray):
        return None
    if isinstance(obj, Enum):
        return obj.value
    if hasattr(obj, "__dataclass_fields__"):
        values = {name: getattr(obj, name) for name in obj.__dataclass_fields__}
        return {k: to_jsonable(v) for k, v in values.items() if not isinstance(v, np.ndarray)}
    if isinstance(obj, dict):
        return {k: to_jsonable(v) for k, v in obj.items() if not isinstance(v, np.ndarray)}
    if isinstance(obj, (list, tuple)):
        return [to_jsonable(v) for v in obj]
    if isinstance(obj, (np.floating,)):
        return float(obj)
    if isinstance(obj, (np.integer,)):
        return int(obj)
    return obj
