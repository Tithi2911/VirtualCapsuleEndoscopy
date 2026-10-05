"""Training-free baseline detector.

Flags compact regions that stand out from the surrounding mucosa in colour
(redness) and surface texture. It exists so the full pipeline - quality gating,
tracking, explanation, missed-finding comparison and reporting - can be built,
demonstrated and tested before a trained model is available. It has NOT been
clinically validated and will both miss lesions and raise false alarms on real
data; replace it with a trained model (see detectors/onnx_seg.py and training/).
"""

from __future__ import annotations

import cv2
import numpy as np

from ..models import Detection, Frame
from .base import components_to_detections


def _robust_z(x: np.ndarray, valid: np.ndarray) -> np.ndarray:
    v = x[valid]
    if v.size == 0:
        return np.zeros_like(x)
    med = np.median(v)
    mad = np.median(np.abs(v - med)) * 1.4826 + 1e-6
    return (x - med) / mad


class HeuristicDetector:
    name = "heuristic-colour-texture"
    version = "0.1.0"

    def __init__(self, saliency_threshold: float = 2.5, min_area: float = 0.002, max_area: float = 0.25):
        self.saliency_threshold = saliency_threshold
        self.min_area = min_area
        self.max_area = max_area

    def saliency(self, image: np.ndarray):
        h, w = image.shape[:2]
        size = max(h, w)
        lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB).astype(np.float32)
        L, a = lab[..., 0], lab[..., 1]

        # Exclude specular glare and the dark lumen / outside the field of view.
        # Glare is dilated by the texture window so its sharp rim does not register as texture.
        k = max(3, int(size / 40) | 1)
        glare = cv2.dilate((L > 240).astype(np.uint8), np.ones((k + 4, k + 4), np.uint8)) > 0
        valid = (L > 35) & ~glare

        # Redness relative to the local background colour.
        bg_sigma = size / 10
        bg_a = cv2.GaussianBlur(a, (0, 0), bg_sigma)
        red_z = _robust_z(a - bg_a, valid)

        # Surface texture / relief: local standard deviation of lightness.
        mean = cv2.blur(L, (k, k))
        sq = cv2.blur(L * L, (k, k))
        local_std = np.sqrt(np.maximum(sq - mean * mean, 0))
        tex_z = _robust_z(local_std, valid)

        # Texture is capped so that sharp edges alone cannot trigger a flag without colour change.
        sal = 0.65 * np.clip(red_z, 0, None) + 0.35 * np.clip(tex_z, 0, 4)
        sal = cv2.GaussianBlur(sal, (0, 0), size / 100)
        sal[~valid] = 0
        return sal, red_z, tex_z, valid

    def detect(self, frame: Frame) -> list[Detection]:
        sal, red_z, tex_z, valid = self.saliency(frame.image)
        heatmap = np.clip(sal / (2 * self.saliency_threshold), 0, 1).astype(np.float32)

        mask = (sal > self.saliency_threshold) & valid
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        mask = cv2.morphologyEx(mask.astype(np.uint8), cv2.MORPH_OPEN, kernel)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel) > 0

        def score(region: np.ndarray):
            contours, _ = cv2.findContours(region.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            area = float(region.sum())
            perimeter = sum(cv2.arcLength(c, True) for c in contours) or 1.0
            compactness = float(min(1.0, 4 * np.pi * area / perimeter**2))
            mean_sal = float(sal[region].mean())
            # Elongated regions are usually folds or vessels, not polyps.
            shape_factor = min(1.0, compactness / 0.6)
            s = 1.0 / (1.0 + np.exp(-1.5 * (mean_sal - self.saliency_threshold - 1.0)))
            return s * shape_factor, {
                "redness_sd": float(red_z[region].mean()),
                "texture_sd": float(tex_z[region].mean()),
                "compactness": compactness,
                "contrast": mean_sal,
            }

        return components_to_detections(
            frame, mask, heatmap, score, "candidate lesion", self.min_area, self.max_area
        )
