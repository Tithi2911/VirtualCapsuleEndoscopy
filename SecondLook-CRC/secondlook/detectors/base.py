"""Detector interface. Every detector must return a per-pixel heatmap with each
detection so the report can explain *why* a region was flagged, not just draw a box."""

from __future__ import annotations

from typing import Protocol

import cv2
import numpy as np

from ..models import Detection, Frame


class Detector(Protocol):
    name: str
    version: str

    def detect(self, frame: Frame) -> list[Detection]: ...


def components_to_detections(
    frame: Frame,
    mask: np.ndarray,
    heatmap: np.ndarray,
    score_fn,
    label: str,
    min_area_fraction: float,
    max_area_fraction: float,
) -> list[Detection]:
    """Split a binary mask into connected regions and score each one.

    `score_fn(region_mask) -> (score, evidence_dict)`.
    """
    h, w = mask.shape
    total = h * w
    n, labels, stats, _ = cv2.connectedComponentsWithStats(mask.astype(np.uint8), connectivity=8)
    detections = []
    for i in range(1, n):
        x, y, bw, bh, area = stats[i]
        if not (min_area_fraction * total <= area <= max_area_fraction * total):
            continue
        region = labels == i
        score, evidence = score_fn(region)
        evidence["area_fraction"] = float(area / total)
        detections.append(
            Detection(
                frame_index=frame.index,
                timestamp_s=frame.timestamp_s,
                bbox=(int(x), int(y), int(bw), int(bh)),
                score=float(score),
                label=label,
                mask=region,
                heatmap=heatmap,
                evidence=evidence,
            )
        )
    return detections
