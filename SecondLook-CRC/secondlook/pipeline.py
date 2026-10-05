"""End-to-end offline analysis: recording in, reviewable findings out."""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Optional

import numpy as np

from . import explain, quality
from .characterise import Characteriser, NullCharacteriser
from .config import AnalysisConfig
from .detectors import Detector
from .ingest import load_frames
from .models import BlindSegment, Finding, Frame, QualityResult, ReportedFinding
from .review import compare
from .tracking import Tracker

FILMSTRIP_FRAMES = 5


@dataclass
class AnalysisResult:
    input_path: str
    config: AnalysisConfig
    detector: str
    detector_version: str
    characteriser: str
    characteriser_version: str
    frames_analysed: int
    frames_informative: int
    duration_s: float
    findings: list[Finding]
    blind_segments: list[BlindSegment]
    unmatched_reported: list[ReportedFinding]
    report_supplied: bool
    timeline: list[tuple[float, bool]]  # (timestamp, informative) per analysed frame
    runtime_s: float
    # finding_id -> {"original", "overlay", "filmstrip", optional "diagnosis"} BGR images
    images: dict[str, dict[str, np.ndarray]] = field(default_factory=dict)


def analyse(
    path: Path | str,
    cfg: AnalysisConfig,
    detector: Detector,
    reported: Optional[list[ReportedFinding]] = None,
    characteriser: Characteriser | None = None,
    image_sequence_fps: float = 1.0,
    progress: Callable[[int], None] | None = None,
) -> AnalysisResult:
    started = time.perf_counter()
    characteriser = characteriser or NullCharacteriser()
    tracker = Tracker(cfg)
    frames: list[Frame] = []  # image dropped unless the frame had detections
    qualities: list[QualityResult] = []
    kept_images: dict[int, np.ndarray] = {}

    for n, frame in enumerate(load_frames(path, cfg.analysis_fps, cfg.working_size, image_sequence_fps)):
        q = quality.assess(frame.image, cfg)
        detections = []
        if q.informative:
            detections = [d for d in detector.detect(frame) if d.score >= cfg.detection_threshold]
        tracker.update(detections, frame.timestamp_s, frame.image.shape)
        if detections:
            kept_images[frame.index] = frame.image
        frames.append(Frame(frame.index, frame.timestamp_s, None, frame.source))
        qualities.append(q)
        if progress and n % 50 == 0:
            progress(n)

    if not frames:
        raise ValueError(f"No frames could be read from {path}")

    quality_by_index = {f.index: q for f, q in zip(frames, qualities)}
    findings = tracker.findings()
    unmatched = compare(findings, reported, cfg)

    images = {}
    for f in findings:
        best = f.best
        img = kept_images[best.frame_index]
        f.rationale = explain.rationale(f, quality_by_index.get(best.frame_index))
        finding_frames = {d.frame_index: kept_images[d.frame_index] for d in f.detections}
        f.characterisation = characteriser.characterise(f, finding_frames)
        picks = np.linspace(0, len(f.detections) - 1, min(FILMSTRIP_FRAMES, len(f.detections))).round().astype(int)
        strip = [kept_images[f.detections[i].frame_index] for i in picks]
        images[f.finding_id] = {
            "original": img,
            "overlay": explain.heatmap_overlay(img, best.heatmap, best.mask),
            "filmstrip": explain.filmstrip(strip),
        }
        c = f.characterisation
        if c.explanation_crop is not None and c.explanation_heatmap is not None:
            images[f.finding_id]["diagnosis"] = explain.diagnosis_overlay(c.explanation_crop, c.explanation_heatmap)

    return AnalysisResult(
        input_path=str(path),
        config=cfg,
        detector=detector.name,
        detector_version=detector.version,
        characteriser=characteriser.name,
        characteriser_version=characteriser.version,
        frames_analysed=len(frames),
        frames_informative=sum(q.informative for q in qualities),
        duration_s=frames[-1].timestamp_s,
        findings=findings,
        blind_segments=quality.blind_segments(frames, qualities, cfg),
        unmatched_reported=unmatched,
        report_supplied=reported is not None,
        timeline=[(f.timestamp_s, q.informative) for f, q in zip(frames, qualities)],
        runtime_s=time.perf_counter() - started,
        images=images,
    )
