"""Visual and written explanation for every flag.

Each finding gets (1) an evidence heatmap blended over the frame, (2) the exact
outline of the flagged region rather than a bounding box, and (3) a plain-English
list of the measurable reasons it was flagged, so a clinician can judge the flag
instead of having to trust it.
"""

from __future__ import annotations

import cv2
import numpy as np

from .models import Finding, QualityResult

OUTLINE_BGR = (255, 255, 0)  # cyan: visible against red/pink mucosa


def heatmap_overlay(image: np.ndarray, heatmap: np.ndarray, mask: np.ndarray | None) -> np.ndarray:
    heat = np.clip(heatmap, 0, 1)
    colour = cv2.applyColorMap((heat * 255).astype(np.uint8), cv2.COLORMAP_INFERNO)
    alpha = (np.clip((heat - 0.15) / 0.85, 0, 1) * 0.45)[..., None]
    out = (image.astype(np.float32) * (1 - alpha) + colour.astype(np.float32) * alpha).astype(np.uint8)
    if mask is not None:
        draw_outline(out, mask)
    return out


def draw_outline(image: np.ndarray, mask: np.ndarray, thickness: int = 2) -> None:
    contours, _ = cv2.findContours(mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    cv2.drawContours(image, contours, -1, (0, 0, 0), thickness + 2)
    cv2.drawContours(image, contours, -1, OUTLINE_BGR, thickness)


def diagnosis_overlay(crop: np.ndarray, heatmap: np.ndarray, size: int = 256) -> np.ndarray:
    """Classifier explanation: which parts of the lesion crop drove the predicted category."""
    crop = cv2.resize(crop, (size, size))
    heat = cv2.resize(np.clip(heatmap, 0, 1).astype(np.float32), (size, size))
    return heatmap_overlay(crop, heat, None)


def _fmt_time(t: float) -> str:
    m, s = divmod(t, 60)
    return f"{int(m):02d}:{s:04.1f}"


def rationale(finding: Finding, best_quality: QualityResult | None) -> list[str]:
    best = finding.best
    ev = best.evidence
    lines = [
        f"In view from {_fmt_time(finding.first.timestamp_s)} to {_fmt_time(finding.last.timestamp_s)} "
        f"({finding.duration_s:.1f} s, {len(finding.detections)} analysed frames); "
        f"strongest at {_fmt_time(best.timestamp_s)} with confidence {best.score:.2f}.",
    ]
    if "redness_sd" in ev:
        lines.append(
            f"Colour: {ev['redness_sd']:+.1f} SD redder than surrounding mucosa "
            "(hyperaemia is a feature of many adenomas and cancers)."
        )
    if "texture_sd" in ev:
        lines.append(f"Surface pattern: {ev['texture_sd']:+.1f} SD more textured/raised than surrounding mucosa.")
    if "compactness" in ev:
        shape = "compact, rounded" if ev["compactness"] > 0.6 else "irregular"
        lines.append(f"Shape: {shape} outline (compactness {ev['compactness']:.2f}; folds and vessels are elongated).")
    if "mean_probability" in ev:
        lines.append(f"Model: mean lesion probability {ev['mean_probability']:.2f} inside the outlined region.")
    if "area_fraction" in ev:
        lines.append(
            f"Apparent size: {100 * ev['area_fraction']:.1f}% of the field of view "
            "(not a size in mm - needs a reference or depth estimate)."
        )
    if best_quality is not None:
        q = "sharp" if best_quality.informative else "degraded (" + ", ".join(best_quality.reasons) + ")"
        lines.append(f"Image quality at best view: {q}.")
    return lines


def filmstrip(images: list[np.ndarray], height: int = 120) -> np.ndarray:
    tiles = []
    for img in images:
        h, w = img.shape[:2]
        tiles.append(cv2.resize(img, (max(1, round(w * height / h)), height)))
        tiles.append(np.full((height, 4, 3), 255, np.uint8))
    return np.hstack(tiles[:-1]) if tiles else np.zeros((height, 1, 3), np.uint8)
