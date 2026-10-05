"""Modality-specific analysis settings.

Traditional colonoscopy and capsule endoscopy produce very different recordings:
colonoscopy is 25-60 fps continuous video under clinician control, while colon
capsules capture 2-35 frames per second over hours with no operator steering,
more debris and stronger vignetting. Each modality therefore gets its own profile.
"""

from __future__ import annotations

from dataclasses import dataclass, replace

from .models import Modality


@dataclass(frozen=True)
class AnalysisConfig:
    modality: Modality
    # Frames analysed per second of recording. Video is subsampled to this rate.
    analysis_fps: float
    # Longest edge frames are resized to before analysis.
    working_size: int
    # Frame quality gates.
    min_sharpness: float
    max_overexposed_fraction: float
    max_underexposed_fraction: float
    # Detector threshold on 0..1 score.
    detection_threshold: float
    # Tracking: link detections into a finding if IoU >= this and the gap <= max_gap_s.
    track_iou: float
    track_max_gap_s: float
    # Discard findings seen in fewer analysed frames than this (suppresses one-frame noise).
    min_track_frames: int
    # A run of non-informative frames at least this long is reported as a blind segment.
    blind_segment_min_s: float
    # Seconds of tolerance when matching a finding to a time in the clinician report.
    report_match_tolerance_s: float


PROFILES: dict[Modality, AnalysisConfig] = {
    Modality.COLONOSCOPY: AnalysisConfig(
        modality=Modality.COLONOSCOPY,
        analysis_fps=5.0,
        working_size=512,
        min_sharpness=40.0,
        max_overexposed_fraction=0.20,
        max_underexposed_fraction=0.55,
        detection_threshold=0.55,
        track_iou=0.2,
        track_max_gap_s=1.0,
        min_track_frames=3,
        blind_segment_min_s=5.0,
        report_match_tolerance_s=5.0,
    ),
    Modality.CAPSULE: AnalysisConfig(
        modality=Modality.CAPSULE,
        # Capsules are already low frame rate; analyse every frame.
        analysis_fps=float("inf"),
        working_size=336,
        min_sharpness=20.0,
        max_overexposed_fraction=0.20,
        max_underexposed_fraction=0.65,
        detection_threshold=0.55,
        # Capsule motion between frames is large and erratic, so tracking is looser.
        track_iou=0.05,
        track_max_gap_s=2.0,
        min_track_frames=2,
        blind_segment_min_s=20.0,
        report_match_tolerance_s=30.0,
    ),
}


def get_config(modality: Modality | str, **overrides) -> AnalysisConfig:
    cfg = PROFILES[Modality(modality)]
    return replace(cfg, **overrides) if overrides else cfg
