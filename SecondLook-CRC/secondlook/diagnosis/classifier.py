"""Run a trained lesion classifier (CADx) exported to ONNX, fully offline.

Each finding is classified from several frames, not only its best one: a lesion
seen from different angles and distances gives a more stable prediction, and
how often the frames agree is itself a useful warning sign. The classifier
abstains rather than guess when the probability is low, the frames disagree,
some frames confidently point to a higher-risk category than the average, or
the model output is not a number, because a confident wrong optical diagnosis
is worse than none. Averaging alone would let a minority of frames showing,
say, a depressed cancerous area vanish into a precancerous call. A benign call
is the one that can lead to a lesion being left in place, so it is withheld
when even one frame confidently gives a neoplastic category.

An occlusion-sensitivity map shows which part of the lesion crop drove the
predicted category, so the reviewer can check it looked at the lesion. The map
measures the fall in the category's log-odds rather than its probability: a
probability close to 100% barely moves when part of the lesion is hidden, which
would leave the map blank exactly when the model is most sure.

The model is described by a JSON sidecar next to the .onnx file, written by
training/train_classifier.py:

    {
      "name": "crc-cadx", "version": "2026.10.0",
      "task": "lesion-classification",
      "classes": ["benign", "precancerous", "cancerous"],
      "input_size": 224, "mean": [...], "std": [...], "crop_margin": 0.25,
      "modality": "colonoscopy", "working_size": 512,
      "temperature": 1.3, "calibrated": true,
      "abstain_below": 0.6, "min_frame_agreement": 0.5,
      "output": "logits"            # or "probabilities"
    }

A sidecar without "modality" is taken to be a colonoscopy model; one without
"calibrated": true has its probabilities labelled as not calibrated.

Input "image": NCHW float32 RGB from preprocess.to_model_input. Output "logits": (N, 3).
"""

from __future__ import annotations

import json
import logging
import math
from collections.abc import Container
from pathlib import Path
from typing import Optional

import cv2
import numpy as np

from ..ingest import file_sha256
from ..models import Characterisation, Detection, Finding
from .metrics import ABSTAIN, MIN_ABSTAIN_BELOW, decide
from .preprocess import DEFAULT_MARGIN, at_working_size, crop_lesion, to_model_input
from .taxonomy import CATEGORIES, DISPLAY_NAME, normalise_label

log = logging.getLogger(__name__)

TASK = "lesion-classification"
SAFETY_STATEMENT = "A prediction of histology, not a histological diagnosis: confirm with histopathology."
EXPLANATION_SIZE = 256  # minimum side of the explanation crop shown in the report
DEFAULT_MODALITY = "colonoscopy"
# Abstain when at least this many frames (and this share of them) each confidently give a
# higher-risk category than the frame average. CATEGORIES is in risk order. A benign call is
# withheld when any single frame confidently gives a neoplastic category: short tracks (2 or 3
# frames) could otherwise never trigger the rule, and a benign call drives "leave in place".
ESCALATION_MIN_FRAMES = 2
ESCALATION_FRACTION = 0.25
OUTPUT_KINDS = ("logits", "probabilities")
PROBABILITY_ATOL = 1e-3


def softmax(logits: np.ndarray, temperature: float = 1.0) -> np.ndarray:
    z = np.asarray(logits, np.float64) / temperature
    z = z - z.max(axis=1, keepdims=True)
    e = np.exp(z)
    return e / e.sum(axis=1, keepdims=True)


def select_detections(detections: list[Detection], available: Container[int], max_frames: int) -> list[Detection]:
    """Up to `max_frames` usable detections in time order: the highest-scoring one from
    each of `max_frames` equal stretches of the track (so always the best detection),
    so the frames cover different views of the lesion rather than one moment."""
    usable = sorted((d for d in detections if d.frame_index in available), key=lambda d: (d.timestamp_s, d.frame_index))
    if len(usable) <= max_frames:
        return usable
    best = max(usable, key=lambda d: d.score)
    picks = []
    for chunk in np.array_split(np.arange(len(usable)), max_frames):
        members = [usable[i] for i in chunk]
        picks.append(best if any(m is best for m in members) else max(members, key=lambda d: d.score))
    return picks


def _as_bgr(image: np.ndarray) -> np.ndarray:
    if image.ndim == 2:
        return cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
    if image.shape[2] == 4:
        return cv2.cvtColor(image, cv2.COLOR_BGRA2BGR)
    return image


def _finite(meta_path: Path, key: str, value, low: float, high: float = math.inf,
            low_open: bool = False, high_open: bool = False) -> float:
    """`value` as a float, or ValueError naming the sidecar key: json accepts NaN, so check."""
    try:
        v = float(value)
    except (TypeError, ValueError):
        v = math.nan
    ok = math.isfinite(v) and (v > low if low_open else v >= low) and (v < high if high_open else v <= high)
    if not ok:
        bounds = f"{'(' if low_open else '['}{low}, {high}{')' if high_open else ']'}"
        raise ValueError(f"{meta_path}: {key} must be a finite number in {bounds}, got {value!r}")
    return v


class OnnxLesionClassifier:
    """Characteriser backed by an ONNX classifier: benign / precancerous / cancerous."""

    # Occlusion positions per explanation at 224 px (a 13 x 13 grid); scaled down
    # for larger inputs so the explanation costs about the same at any size.
    max_occlusion_positions = 169
    occlusion_batch = 64

    def __init__(
        self,
        model_path: Path | str,
        providers: list[str] | None = None,
        max_frames: int = 8,
        explain: bool = True,
    ):
        try:
            import onnxruntime as ort
        except ImportError as e:  # pragma: no cover - optional dependency
            raise ImportError("Install the 'model' extra: pip install -e '.[model]'") from e

        model_path = Path(model_path)
        if not model_path.exists():
            raise FileNotFoundError(f"Lesion classifier not found: {model_path}")
        meta_path = model_path.with_suffix(".json")
        if not meta_path.exists():
            raise FileNotFoundError(
                f"Lesion classifier sidecar not found: {meta_path}. It defines the class order and "
                "calibration; training/train_classifier.py writes it next to the .onnx file."
            )
        self.meta = json.loads(meta_path.read_text())
        if self.meta.get("task") != TASK:
            raise ValueError(f"{meta_path} is not a lesion classifier (task={self.meta.get('task')!r}, expected {TASK!r})")
        classes = [normalise_label(c) for c in self.meta.get("classes", [])]
        if len(classes) != len(CATEGORIES) or sorted(classes) != sorted(CATEGORIES):
            raise ValueError(f"{meta_path}: classes {self.meta.get('classes')!r} must be a permutation of {CATEGORIES}")
        # Model output column for each category, in CATEGORIES order.
        self.class_order = np.array([classes.index(c) for c in CATEGORIES])

        def get(key, default):
            value = self.meta.get(key)
            return default if value is None else value

        self.name = str(get("name", model_path.stem))
        self.version = str(get("version", "unversioned"))
        self.input_size = int(_finite(meta_path, "input_size", get("input_size", 224), 8))
        self.mean = np.array(get("mean", [0.485, 0.456, 0.406]), np.float32)
        self.std = np.array(get("std", [0.229, 0.224, 0.225]), np.float32)
        if self.mean.shape != (3,) or self.std.shape != (3,) or not (np.isfinite(self.mean).all()
                                                                     and np.isfinite(self.std).all()
                                                                     and (self.std > 0).all()):
            raise ValueError(f"{meta_path}: mean and std must be three finite numbers each, std > 0")
        self.crop_margin = _finite(meta_path, "crop_margin", get("crop_margin", DEFAULT_MARGIN), 0, 10)
        self.temperature = _finite(meta_path, "temperature", get("temperature", 1.0), 0, low_open=True)
        # Only an explicit "calibrated": true from a successful temperature fit counts.
        self.calibrated = self.meta.get("calibrated") is True and "temperature" in self.meta
        self.abstain_below = _finite(meta_path, "abstain_below", get("abstain_below", 0.6), MIN_ABSTAIN_BELOW, 1,
                                     high_open=True)
        self.min_frame_agreement = _finite(meta_path, "min_frame_agreement", get("min_frame_agreement", 0.5), 0, 1)
        output = str(get("output", "logits")).strip().lower()
        if output not in OUTPUT_KINDS:
            # Any other value would silently read logits as probabilities, or the reverse.
            raise ValueError(f"{meta_path}: output must be one of {list(OUTPUT_KINDS)}, got {self.meta.get('output')!r}")
        self.output_is_logits = output == "logits"
        modality = get("modality", DEFAULT_MODALITY)
        self.modalities = [str(m) for m in (modality if isinstance(modality, list) else [modality])]
        working_size = self.meta.get("working_size")
        self.working_size = None if working_size is None else int(_finite(meta_path, "working_size", working_size, 32))
        self.max_frames = max(1, int(max_frames))
        self.explain = explain

        self.session = ort.InferenceSession(str(model_path), providers=providers or ort.get_available_providers())
        inputs = self.session.get_inputs()
        names = [i.name for i in inputs]
        self.input_name = "image" if "image" in names else names[0]
        outputs = self.session.get_outputs()
        out_names = [o.name for o in outputs]
        self.output_name = "logits" if "logits" in out_names else out_names[0]
        in_shape = list(inputs[names.index(self.input_name)].shape)
        batch_dim = in_shape[0] if in_shape else None
        # Models exported with a fixed batch size are run in chunks of that size, the last one padded.
        self.batch_limit = batch_dim if isinstance(batch_dim, int) and batch_dim > 0 else None
        self._check_model(meta_path, in_shape, list(outputs[out_names.index(self.output_name)].shape))

        self.sha256 = file_sha256(model_path)
        self.sidecar_sha256 = file_sha256(meta_path)
        self.info = self._describe()

    def _check_model(self, meta_path: Path, in_shape: list, out_shape: list) -> None:
        """Fail at load time, not on every finding, if the model does not fit its sidecar."""
        s, k = self.input_size, len(CATEGORIES)
        expected = [3, s, s]
        if len(in_shape) != 4 or any(isinstance(d, int) and d != e for d, e in zip(in_shape[1:], expected)):
            raise ValueError(f"{meta_path}: the model input has shape {in_shape}; the sidecar needs "
                             f"(batch, 3, {s}, {s}) (input_size {s})")
        if len(out_shape) != 2 or (isinstance(out_shape[1], int) and out_shape[1] != k):
            raise ValueError(f"{meta_path}: the model output has shape {out_shape}; expected (batch, {k}), "
                             f"one column per class in {self.meta.get('classes')!r}")
        n = self.batch_limit or 1
        try:
            out = self.session.run([self.output_name], {self.input_name: np.zeros((n, 3, s, s), np.float32)})[0]
        except Exception as e:
            raise ValueError(f"{meta_path}: the model does not run on a ({n}, 3, {s}, {s}) input: {e}") from e
        if out.shape != (n, k):
            raise ValueError(f"{meta_path}: the model returned shape {out.shape} for a batch of {n}; expected ({n}, {k})")
        if not np.isfinite(out).all():
            raise ValueError(f"{meta_path}: the model returns non-finite values (NaN or infinity) on a plain input")
        if not self.output_is_logits and not (
                (out >= -PROBABILITY_ATOL).all() and (out <= 1 + PROBABILITY_ATOL).all()
                and np.allclose(out.sum(axis=1), 1.0, atol=PROBABILITY_ATOL)):
            raise ValueError(f"{meta_path}: the sidecar says output is 'probabilities', but the model returned "
                             f"{out[0].round(4).tolist()}, which is not a probability per class summing to 1; "
                             "for raw scores set output to 'logits'")
        if self.batch_limit and self.batch_limit > 1:
            try:  # a partial batch, as the last chunk of frames and every occlusion map produce
                self._run(np.zeros((1, 3, s, s), np.float32))
            except Exception as e:
                raise ValueError(f"{meta_path}: the model does not run on a partial batch: {e}") from e

    def _describe(self) -> dict:
        """Provenance and validation status shown with every AI category in the report."""
        meta = self.meta
        status = []
        if meta.get("synthetic_training_data") is True:
            status.append("Trained on synthetic images only: a software test, not a clinical model.")
        test_n = ((meta.get("metrics") or {}).get("test") or {}).get("n")
        status.append(f"Internal test set: {test_n} images held out from its training sources." if test_n
                      else "No test results are recorded with the model.")
        external = meta.get("external_validation")
        status.append(f"External validation recorded: {external}." if external
                      else "No external validation is recorded with the model.")
        if not self.calibrated:
            status.append("Probabilities are not calibrated.")
        data = meta.get("training_data") or []
        init_data = [d for d in meta.get("init_training_datasets") or [] if isinstance(d, dict) and d.get("path")]
        return {
            "sha256": self.sha256,
            "sidecar_sha256": self.sidecar_sha256,
            "calibrated": self.calibrated,
            "temperature": self.temperature,
            "modalities": self.modalities,
            "working_size": self.working_size,
            "abstain_below": self.abstain_below,
            "training_data": [Path(str(p)).name for p in (data if isinstance(data, list) else [data])],
            # Data an --init checkpoint (and its own ancestors) was trained on before fine-tuning.
            "init_training_data": [Path(str(d["path"])).name for d in init_data],
            "synthetic_training_data": meta.get("synthetic_training_data"),
            "validation_status": " ".join(status),
            "validation_warnings": [str(w) for w in meta.get("validation_warnings") or []],
            "intended_use": str(meta.get("intended_use") or "No intended-use statement in the model file."),
        }

    def unsupported_reason(self, cfg) -> Optional[str]:
        """Why this model must not characterise a recording analysed with `cfg`, or None."""
        modality = getattr(cfg.modality, "value", cfg.modality)
        if modality not in self.modalities:
            return (f"the model was trained for {' and '.join(self.modalities)} images and is not "
                    f"validated for {modality}")
        if self.working_size is not None and cfg.working_size < self.working_size:
            return (f"frames are analysed at {cfg.working_size} px but the model was trained on images "
                    f"shrunk to {self.working_size} px, so lesions would reach it with less detail than in training")
        return None

    def lesion_crop(self, image: np.ndarray, bbox, size: int | None = None) -> np.ndarray:
        """The crop the model sees: the frame at the training working size, then crop_lesion."""
        image, bbox = at_working_size(_as_bgr(image), bbox, self.working_size)
        return crop_lesion(image, bbox, size or self.input_size, self.crop_margin)

    def _run(self, x: np.ndarray) -> np.ndarray:
        """Raw model output (N, 3) in the model's own class order. A model with a fixed batch size
        is run in chunks of that size, the last one padded with zeros and the padding dropped."""
        step = self.batch_limit or max(len(x), 1)
        outs = []
        for i in range(0, len(x), step):
            chunk = x[i : i + step]
            n = len(chunk)
            if n < step:
                chunk = np.concatenate([chunk, np.zeros((step - n, *chunk.shape[1:]), chunk.dtype)])
            out = self.session.run([self.output_name], {self.input_name: chunk})[0]
            if len(out) != len(chunk):
                raise ValueError(f"the model returned {len(out)} rows for a batch of {len(chunk)}")
            outs.append(out[:n])
        return np.concatenate(outs) if outs else np.zeros((0, len(CATEGORIES)))

    def logits(self, x: np.ndarray) -> np.ndarray:
        """Uncalibrated logits (N, 3) in CATEGORIES order for a to_model_input batch
        (log-probabilities for a model that outputs probabilities)."""
        out = self._run(x).astype(np.float64)[:, self.class_order]
        return out if self.output_is_logits else np.log(np.clip(out, 1e-12, None))

    def predict(self, x: np.ndarray) -> np.ndarray:
        """Temperature-scaled probabilities (N, 3) in CATEGORIES order for a to_model_input batch."""
        return softmax(self.logits(x), self.temperature)

    def log_odds(self, x: np.ndarray, category_index: int) -> np.ndarray:
        """Temperature-scaled log-odds log(p / (1 - p)) of one category, computed stably from the logits."""
        z = self.logits(x) / self.temperature
        others = np.delete(z, category_index, axis=1)
        m = others.max(axis=1, keepdims=True)
        return z[:, category_index] - (m[:, 0] + np.log(np.exp(others - m).sum(axis=1)))

    def characterise(self, finding: Finding, frames: dict[int, np.ndarray]) -> Characterisation:
        try:
            return self._characterise(finding, frames)
        except Exception as e:  # one bad finding must not stop the whole analysis
            log.warning("Lesion classifier failed on %s: %s", finding.finding_id, e)
            return self._abstain(f"classifier error ({type(e).__name__}: {e})")

    def _abstain(self, reason: str, **fields) -> Characterisation:
        return Characterisation(
            model=self.name,
            model_version=self.version,
            abstained=True,
            abstain_reason=reason,
            note=f"No AI category given: {reason}. {SAFETY_STATEMENT}",
            **fields,
        )

    def _characterise(self, finding: Finding, frames: dict[int, np.ndarray]) -> Characterisation:
        crops, used = [], []
        for d in select_detections(finding.detections, frames, self.max_frames):
            try:
                crops.append(self.lesion_crop(frames[d.frame_index], d.bbox))
                used.append(d)
            except (cv2.error, ValueError, IndexError):
                continue
        if not crops:
            return self._abstain("no usable frames")

        probs = self.predict(to_model_input(crops, self.mean, self.std))
        finite = np.isfinite(probs).all(axis=1)
        n_bad = int((~finite).sum())
        if n_bad == len(crops):
            return self._abstain(f"the model returned non-finite output (NaN or infinity) for all {n_bad} frames")
        crops = [c for c, ok in zip(crops, finite) if ok]
        used = [d for d, ok in zip(used, finite) if ok]
        probs = probs[finite]

        n = len(crops)
        agg = probs.mean(axis=0)
        top = int(agg.argmax())
        agree = int((probs.argmax(axis=1) == top).sum())
        confidence = float(agg[top])
        cancer = CATEGORIES.index("cancerous")
        neo = float(agg[CATEGORIES.index("precancerous")] + agg[cancer])
        fields = dict(
            probabilities={c: float(p) for c, p in zip(CATEGORIES, agg)},
            calibrated=self.calibrated,
            confidence=confidence,
            frames_used=n,
            frame_agreement=agree / n,
            malignancy_risk=float(agg[cancer]),
            peak_malignancy_risk=float(probs[:, cancer].max()),
            neoplasia_risk=neo,
        )

        reasons = []
        if n_bad:
            reasons.append(f"the model returned non-finite output for {n_bad} of {n + n_bad} frames")
        if confidence < self.abstain_below:
            reasons.append(f"low confidence ({confidence:.2f} < {self.abstain_below:.2f})")
        elif decide(agg[None], self.abstain_below)[0] == ABSTAIN:
            reasons.append(f"not called benign because P(precancerous) + P(cancerous) = {neo:.2f} "
                           f"is not below P(benign) = {confidence:.2f}")
        if agree / n < self.min_frame_agreement:
            reasons.append(f"frames disagree ({agree}/{n} agree)")
        frame_calls = decide(probs, self.abstain_below)
        higher = frame_calls > top  # abstaining frames are ABSTAIN (-1), never higher
        needed = 1 if CATEGORIES[top] == "benign" else max(ESCALATION_MIN_FRAMES, math.ceil(ESCALATION_FRACTION * n))
        if higher.sum() >= needed:
            worst = CATEGORIES[int(frame_calls.max())]
            reasons.append(f"{int(higher.sum())} of {n} frames confidently suggest a higher-risk category "
                           f"({DISPLAY_NAME[worst]})")
        if reasons:
            return self._abstain("; ".join(reasons), **fields)

        c = Characterisation(
            model=self.name,
            model_version=self.version,
            category=CATEGORIES[top],
            note=f"AI optical diagnosis from {n} frame{'s' if n != 1 else ''} ({agree}/{n} agree). {SAFETY_STATEMENT}",
            **fields,
        )
        if self.explain:
            b = max(range(n), key=lambda i: used[i].score)
            heat = self.occlusion_map(crops[b], top)
            side = max(self.input_size, EXPLANATION_SIZE)
            if side == self.input_size:
                c.explanation_crop = crops[b]
            else:  # same square region at display resolution, so small model inputs are not shown blurred
                best = used[b]
                c.explanation_crop = self.lesion_crop(frames[best.frame_index], best.bbox, side)
                heat = cv2.resize(heat, (side, side), interpolation=cv2.INTER_LINEAR)
            c.explanation_heatmap = heat
        return c

    def _occlusion_grid(self) -> tuple[int, list[int]]:
        """Patch size and start offsets (same for both axes) covering the whole crop."""
        s = self.input_size
        patch = max(1, s // 7)
        stride = max(1, patch // 2)
        budget = max(25, min(self.max_occlusion_positions, self.max_occlusion_positions * 224**2 // s**2))
        if ((s - patch) // stride + 1) ** 2 > budget:
            k = math.isqrt(budget)
            patch = math.ceil(2 * s / (k + 1))  # fewer, larger patches, still half-overlapping
            stride = max(1, patch // 2)
        starts = list(range(0, s - patch + 1, stride))
        if starts[-1] != s - patch:
            starts.append(s - patch)
        return patch, starts

    @staticmethod
    def _border_fill(x: np.ndarray) -> np.ndarray:
        """Per-channel median of the crop's outer border (1/16 of its side), shape (3, 1, 1).
        With the crop margin around the lesion that border is mostly surrounding mucosa."""
        s = x.shape[-1]
        r = max(1, s // 16)
        img = x[0]
        ring = np.concatenate([img[:, :r].reshape(3, -1), img[:, -r:].reshape(3, -1),
                               img[:, r:-r, :r].reshape(3, -1), img[:, r:-r, -r:].reshape(3, -1)], axis=1)
        return np.median(ring, axis=1).astype(np.float32)[:, None, None]

    def occlusion_map(self, crop: np.ndarray, category_index: int) -> np.ndarray:
        """0..1 map (input_size x input_size): how much hiding each region lowers the
        log-odds of the category, i.e. makes the model less sure of it.

        Occluded patches are filled with the median colour of the crop's outer border,
        so they look like featureless surrounding mucosa rather than a grey or black
        hole the model never saw in training. Log-odds scale with 1/temperature, so
        after normalisation the map does not depend on calibration.
        """
        s = self.input_size
        x = to_model_input([crop], self.mean, self.std)
        fill = self._border_fill(x)
        ref = float(self.log_odds(x, category_index)[0])
        patch, starts = self._occlusion_grid()
        positions = [(y, x0) for y in starts for x0 in starts]

        drop = np.empty(len(positions), np.float64)
        for i in range(0, len(positions), self.occlusion_batch):
            chunk = positions[i : i + self.occlusion_batch]
            batch = np.repeat(x, len(chunk), axis=0)
            for j, (y, x0) in enumerate(chunk):
                batch[j, :, y : y + patch, x0 : x0 + patch] = fill
            drop[i : i + len(chunk)] = np.maximum(0.0, ref - self.log_odds(batch, category_index))
        drop[~np.isfinite(drop)] = 0.0

        heat = np.zeros((s, s), np.float64)
        count = np.zeros((s, s), np.float64)
        for (y, x0), d in zip(positions, drop):
            heat[y : y + patch, x0 : x0 + patch] += d
            count[y : y + patch, x0 : x0 + patch] += 1
        heat /= np.maximum(count, 1)
        if heat.max() <= 0:
            return np.zeros((s, s), np.float32)
        heat = cv2.GaussianBlur(heat.astype(np.float32), (0, 0), sigmaX=max(1.0, patch / 2))
        return (heat / heat.max()).astype(np.float32) if heat.max() > 0 else np.zeros((s, s), np.float32)
