"""Run a trained polyp segmentation model exported to ONNX, fully offline.

ONNX keeps the runtime small and vendor-neutral (CPU, CUDA, OpenVINO, CoreML
execution providers) and lets the same model later be embedded in MADSyncro or
MADAlpha. The model is described by a JSON sidecar next to the .onnx file:

    {
      "name": "unet-kvasir-seg",
      "version": "2026.10.0",
      "input_size": 352,
      "mean": [0.485, 0.456, 0.406],
      "std":  [0.229, 0.224, 0.225],
      "threshold": 0.5,
      "label": "polyp",
      "output": "logits"            # or "probabilities"
    }

Input: NCHW float32 RGB. Output: (1, 1, H, W) map. training/train_segmentation.py
produces models in exactly this format.
"""

from __future__ import annotations

import json
from pathlib import Path

import cv2
import numpy as np

from ..models import Detection, Frame
from .base import components_to_detections


class OnnxSegmentationDetector:
    def __init__(self, model_path: Path | str, providers: list[str] | None = None):
        try:
            import onnxruntime as ort
        except ImportError as e:  # pragma: no cover - optional dependency
            raise ImportError("Install the 'model' extra: pip install -e '.[model]'") from e

        model_path = Path(model_path)
        meta_path = model_path.with_suffix(".json")
        self.meta = json.loads(meta_path.read_text()) if meta_path.exists() else {}
        self.name = self.meta.get("name", model_path.stem)
        self.version = str(self.meta.get("version", "unversioned"))
        self.input_size = int(self.meta.get("input_size", 352))
        self.mean = np.array(self.meta.get("mean", [0.485, 0.456, 0.406]), np.float32)
        self.std = np.array(self.meta.get("std", [0.229, 0.224, 0.225]), np.float32)
        self.threshold = float(self.meta.get("threshold", 0.5))
        self.label = self.meta.get("label", "polyp")
        self.output_is_logits = self.meta.get("output", "logits") == "logits"
        self.session = ort.InferenceSession(
            str(model_path), providers=providers or ort.get_available_providers()
        )
        self.input_name = self.session.get_inputs()[0].name

    def _prob_map(self, image: np.ndarray) -> np.ndarray:
        h, w = image.shape[:2]
        rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        x = cv2.resize(rgb, (self.input_size, self.input_size)).astype(np.float32) / 255.0
        x = ((x - self.mean) / self.std).transpose(2, 0, 1)[None]
        out = self.session.run(None, {self.input_name: x})[0][0, 0]
        if self.output_is_logits:
            out = 1.0 / (1.0 + np.exp(-out))
        return cv2.resize(out.astype(np.float32), (w, h))

    def detect(self, frame: Frame) -> list[Detection]:
        prob = self._prob_map(frame.image)
        mask = prob >= self.threshold

        def score(region: np.ndarray):
            return float(prob[region].max()), {"mean_probability": float(prob[region].mean())}

        return components_to_detections(frame, mask, prob, score, self.label, 0.0005, 0.6)
