"""Train the lesion classifier (benign / precancerous / cancerous) and export it for SecondLook.

The model performs optical diagnosis. From an endoscopic image of a detected
lesion it predicts the histology category that a pathologist would later
assign. It sees tissue surface and vessels, not cells, so histopathology stays
the reference standard. A model trained here is a research prototype and not a
medical device.

Output (in --out): model.onnx + model.json (sidecar read by
secondlook.diagnosis.classifier), best.pt (PyTorch checkpoint for fine-tuning),
model.development.json (hashed development patients and images, for
--evaluate-only), metrics.json, predictions_test.csv and split.json.

Choices that keep the reported numbers honest:

* Images are shrunk to the analysis working size of --modality (512 px for
  colonoscopy, as secondlook does to every analysed frame) before the lesion is
  cropped, so the model learns from the detail it will actually get in use.
* Inverse-frequency class weights help the rare classes train, but they shift the
  model's class prior. The shift is removed from the exported logits (logit
  adjustment) before the temperature is fitted, so probabilities track the real
  class mix rather than inflating the rare cancer class.
* With --init, patients and images the checkpoint was trained on (recorded,
  hashed, in best.pt) are kept in the training split, so they cannot inflate
  val/test. Images without a patient_id are identified by their content, so this
  holds when the data set is moved or copied.
* The sidecar records resolved data paths and labels.csv hashes (including those
  of an --init checkpoint's data); model.development.json records salted hashes
  of every development patient ID and image file. --evaluate-only uses both to
  detect overlap with the development data.

Distribute only model.onnx and model.json. best.pt, model.development.json,
split.json and predictions_test.csv are derived from patient data (split.json and
the predictions name pseudonymous patient IDs; salted hashes of short IDs can be
reversed by trying every possible ID) and stay with the data owner.

Dataset layout. Each --data directory has a labels.csv:

    image,label,patient_id,mask,x,y,w,h,split
    images/p001_1.png,tubular adenoma,P001,masks/p001_1.png,,,,,
    images/p002_1.png,hyperplastic polyp,P002,,120,80,64,60,
    images/p003_1.png,adenocarcinoma,P003,,,,,,test

  image       path relative to the directory
  label       histology or category text, mapped by secondlook.diagnosis.taxonomy.category_for
  patient_id  strongly recommended: the split is made by patient, so frames of one
              patient never appear in both training and test
  mask | x,y,w,h  lesion location (optional; without it the whole image is used)
  split       optional train/val/test; overrides the automatic 70/15/15 patient split

Typical workflow:

    # 1. Pre-train on procedural synthetic lesions. This teaches the pipeline, not pathology.
    python training/make_synthetic_lesions.py --out data/synth_lesions --n-per-class 400
    python training/train_classifier.py --data data/synth_lesions --out models/cadx-synth \\
        --backbone resnet18 --weights weights/resnet18-f37072fd.pth --epochs 15

    # 2. Fine-tune on real histology-labelled lesions (e.g. PICCOLO, SUN, REAL-Colon, local data).
    python training/train_classifier.py --data data/piccolo --data data/centre_a \\
        --backbone resnet18 --init models/cadx-synth/best.pt --out models/cadx-v1 \\
        --name cadx-resnet18 --version 2026.10.0

    # 3. External validation on a centre or device never used for training.
    python training/train_classifier.py --evaluate-only models/cadx-v1/model.onnx \\
        --data data/centre_b --out reports/cadx-v1-centre_b

    # 4. Use it.
    secondlook analyse procedure.mp4 --classifier models/cadx-v1/model.onnx

This machine may be offline. --pretrained downloads torchvision ImageNet weights.
Without network access, copy the .pth file over and pass --weights PATH instead.
Check each dataset's licence before any commercial use.
"""

from __future__ import annotations

import argparse
import contextlib
import csv
import hashlib
import inspect
import json
import math
import random
import secrets
import sys
import time
import warnings
from collections import Counter
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import NoReturn, Optional

import cv2
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset

from secondlook.config import get_config
from secondlook.diagnosis.metrics import (
    ABSTAIN,
    MIN_ABSTAIN_BELOW,
    bootstrap_ci,
    classification_report,
    decide,
    expected_calibration_error,
)
from secondlook.diagnosis.preprocess import (
    DEFAULT_MARGIN,
    at_working_size,
    bbox_from_mask,
    crop_lesion,
    to_model_input,
)
from secondlook.diagnosis.taxonomy import CATEGORIES, category_for
from secondlook.ingest import file_sha256
from secondlook.models import Modality

MEAN = [0.485, 0.456, 0.406]
STD = [0.229, 0.224, 0.225]
BACKBONES = ("tiny", "resnet18", "resnet50", "efficientnet_b0", "convnext_tiny")
HEAD_PREFIX = {"tiny": "head.", "resnet18": "fc.", "resnet50": "fc.", "efficientnet_b0": "classifier.",
               "convnext_tiny": "classifier."}
SPLITS = ("train", "val", "test")
SPLIT_ALIASES = {"train": "train", "training": "train", "val": "val", "valid": "val", "validation": "val",
                 "dev": "val", "test": "test", "testing": "test", "holdout": "test"}
LABEL_SMOOTHING = 0.05
GRAD_CLIP = 1.0
BBOX_JITTER = 0.10
TEMPERATURE_RANGE = (0.05, 20.0)
MAX_CROP_MARGIN = 10.0  # the runtime classifier refuses larger margins
MIN_INPUT_SIZE = 8
DEVELOPMENT_SUFFIX = ".development.json"
ONNX_ATOL = 1e-3
MIN_RELIABLE_PER_CLASS = 30
INTENDED_USE = (
    "Research use only. Not a medical device and not validated for clinical use. Predicts the likely "
    "histology category (benign, precancerous or cancerous) of a detected lesion from endoscopic images "
    "(optical diagnosis); it does not examine cells. Predictions must not guide treatment on their own "
    "and require histopathological confirmation, which remains the reference standard."
)
PIVI_NOTE = (
    "Reference point: the ASGE PIVI benchmark for 'diagnose-and-leave' of diminutive (<= 5 mm) "
    "rectosigmoid polyps is an NPV >= 90% for adenomatous histology, using high-confidence optical "
    "diagnoses. This data set is not restricted to such polyps, so this figure is not evidence of "
    "meeting that benchmark."
)


def fail(message: str) -> NoReturn:
    raise SystemExit(f"ERROR: {message}")


def _n(count: int, noun: str) -> str:
    return f"{count} {noun}{'' if count == 1 else 's'}"


# ----------------------------------------------------------------------------- data


@dataclass
class Sample:
    path: Path  # resolved image path
    image: str  # image path as written in labels.csv
    dataset: str  # the --data directory it came from
    label: str  # label as written in labels.csv
    category: int  # index into CATEGORIES
    patient_id: str  # "" when not given
    # Split unit: the normalised patient_id, or the image content ("image:<sha256>") when no
    # patient_id is given, so the unit does not change when the data set is moved or copied.
    group: str
    bbox: Optional[tuple[int, int, int, int]]
    bbox_shape: Optional[tuple[int, int]]  # (h, w) of the mask the bbox came from, if any
    split: Optional[str]
    image_sha256: str = ""  # SHA-256 of the image file's bytes

    def bbox_in(self, shape: tuple[int, ...]) -> Optional[tuple[int, int, int, int]]:
        """The bbox in the pixel grid of an image of `shape` (masks may be stored at another size)."""
        if self.bbox is None:
            return None
        h, w = shape[:2]
        x, y, bw, bh = self.bbox
        if self.bbox_shape is not None and self.bbox_shape != (h, w):
            sy, sx = h / self.bbox_shape[0], w / self.bbox_shape[1]
            x, y, bw, bh = x * sx, y * sy, bw * sx, bh * sy
        x0, y0 = max(0, int(round(x))), max(0, int(round(y)))
        x1, y1 = min(w, int(round(x + bw))), min(h, int(round(y + bh)))
        if x1 <= x0 or y1 <= y0:
            raise ValueError(f"lesion box {self.bbox} lies outside image {self.path} ({w}x{h})")
        return x0, y0, x1 - x0, y1 - y0


def _resolve(root: Path, value: str) -> Path:
    p = Path(value)
    return p if p.is_absolute() else root / p


def _read_bbox(row: dict[str, str]) -> Optional[tuple[int, int, int, int]]:
    values = [row.get(k, "") for k in ("x", "y", "w", "h")]
    if not any(values):
        return None
    if not all(values):
        raise ValueError("give all of x, y, w, h or none of them")
    x, y, w, h = (int(round(float(v))) for v in values)
    if w <= 0 or h <= 0:
        raise ValueError(f"bbox width and height must be positive, got w={w} h={h}")
    return x, y, w, h


def image_shape(path: Path) -> tuple[int, int]:
    """(height, width) as cv2.imread will see the image, from the file header where possible,
    so every row can be checked before training starts."""
    try:
        from PIL import Image

        with Image.open(path) as im:
            w, h = im.size
            if im.getexif().get(0x0112) in (5, 6, 7, 8):  # EXIF orientation that swaps the axes
                w, h = h, w
            return h, w
    except Exception:  # no Pillow, or a format it cannot read: decode with OpenCV instead
        image = cv2.imread(str(path), cv2.IMREAD_COLOR)
        if image is None:
            raise ValueError(f"cannot read image {path}") from None
        return image.shape[:2]


def _mask_bbox(path: Path) -> tuple[Optional[tuple[int, int, int, int]], tuple[int, int]]:
    mask = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    if mask is None:
        raise ValueError(f"cannot read mask {path}")
    # JPEG masks carry compression noise near 0, so threshold at mid-grey unless the mask is 0/1.
    threshold = 127 if mask.max() > 1 else 0
    return bbox_from_mask(mask > threshold), mask.shape[:2]


def patient_key(patient_id: str) -> str:
    """A patient ID as compared everywhere (split, --init, overlap): case and outer spaces ignored."""
    return patient_id.strip().casefold()


def identity_keys(s: Sample) -> set[str]:
    """What identifies a sample's source: its patient (when given) and its image content."""
    return {s.group, f"image:{s.image_sha256}"} if s.image_sha256 else {s.group}


def lookup_keys(s: Sample) -> set[str]:
    """identity_keys plus the raw patient ID, which checkpoints of earlier versions hashed."""
    return identity_keys(s) | ({s.patient_id} if s.patient_id else set())


def load_samples(data_dirs: list[Path]) -> tuple[list[Sample], list[str]]:
    """Read every labels.csv. Collects every problem (unknown labels, missing or unreadable files,
    bad boxes or splits) and reports them together, so the user can fix them in one go."""
    samples: list[Sample] = []
    notes: list[str] = []
    unknown: Counter[str] = Counter()
    missing: list[str] = []
    problems: list[str] = []
    empty_masks = 0
    for root in data_dirs:
        csv_path = root / "labels.csv"
        if not csv_path.is_file():
            fail(f"{csv_path} not found. Each --data directory needs a labels.csv (see --help).")
        with csv_path.open(newline="", encoding="utf-8-sig") as fh:
            reader = csv.DictReader(fh)
            columns = {(c or "").strip().lower() for c in reader.fieldnames or []}
            for required in ("image", "label"):
                if required not in columns:
                    fail(f"{csv_path} has no '{required}' column (columns: {sorted(columns)}).")
            if "patient_id" not in columns:
                notes.append(f"{csv_path} has no patient_id column.")
            for line, raw in enumerate(reader, start=2):
                row = {(k or "").strip().lower(): (v or "").strip() for k, v in raw.items() if isinstance(v, str)}
                where = f"{csv_path}:{line}"
                ok = True  # every check runs on every row, so all problems are listed at once
                category = -1
                try:
                    category = CATEGORIES.index(category_for(row.get("label", "")).value)
                except KeyError:
                    unknown[row.get("label", "")] += 1
                    ok = False
                if not row.get("image"):
                    problems.append(f"{where}: empty image path")
                    continue
                path = _resolve(root, row["image"])
                image_ok = path.is_file()
                if not image_ok:
                    missing.append(str(path))
                    ok = False
                split = None
                if row.get("split"):
                    split = SPLIT_ALIASES.get(row["split"].lower())
                    if split is None:
                        problems.append(f"{where}: split must be train, val or test, got {row['split']!r}")
                        ok = False
                bbox, bbox_shape = None, None
                try:
                    bbox = _read_bbox(row)
                    if bbox is None and row.get("mask"):
                        mask_path = _resolve(root, row["mask"])
                        if not mask_path.is_file():
                            missing.append(str(mask_path))
                            ok = False
                        else:
                            bbox, bbox_shape = _mask_bbox(mask_path)
                            empty_masks += bbox is None
                except ValueError as e:
                    problems.append(f"{where}: {e}")
                    ok = False
                patient = row.get("patient_id", "")
                sample = Sample(
                    path=path, image=row["image"], dataset=str(root), label=row.get("label", ""),
                    category=category, patient_id=patient, group="", bbox=bbox, bbox_shape=bbox_shape, split=split,
                )
                if image_ok:
                    try:  # unreadable images and boxes outside the image, before hours of training
                        sample.bbox_in(image_shape(path))
                    except ValueError as e:
                        problems.append(f"{where}: {e}")
                        ok = False
                if not ok:
                    continue
                # Content, not path: a moved or copied data set keeps its groups and hashes.
                sample.image_sha256 = file_sha256(path)
                sample.group = patient_key(patient) if patient_key(patient) else f"image:{sample.image_sha256}"
                samples.append(sample)
    report = []
    if unknown:
        listed = ", ".join(f"{label!r} ({_n(n, 'row')})" for label, n in unknown.most_common())
        report.append(f"Unknown labels: {listed}. Fix labels.csv or add them to HISTOLOGY_TO_CATEGORY in "
                      "secondlook/diagnosis/taxonomy.py.")
    if missing:
        report.append(f"{_n(len(missing), 'referenced file')} not found, e.g.:\n  " + "\n  ".join(missing[:10]))
    if problems:
        report.append(f"{_n(len(problems), 'problem')} in labels.csv:\n  " + "\n  ".join(problems[:20]))
    if report:
        fail("\n".join(report))
    if not samples:
        fail("No labelled images found.")
    if empty_masks:
        notes.append(f"{empty_masks} masks are empty; those images are used whole.")
    return samples, notes


def read_image(path: Path) -> np.ndarray:
    image = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if image is None:
        raise ValueError(f"cannot read image {path}")
    return image


def jitter_view(image: np.ndarray, bbox, rng=random):
    """Perturb the lesion box by up to +-10% (shift and scale), as a real detector's boxes vary.

    Rows without a box get a mild random crop instead, so their framing varies too.
    """
    h, w = image.shape[:2]
    if bbox is None:
        s = rng.uniform(0.85, 1.0)
        cw, ch = max(8, int(round(w * s))), max(8, int(round(h * s)))
        x0, y0 = rng.randint(0, max(0, w - cw)), rng.randint(0, max(0, h - ch))
        return image[y0:y0 + ch, x0:x0 + cw], None
    x, y, bw, bh = bbox
    cx = x + bw / 2 + rng.uniform(-BBOX_JITTER, BBOX_JITTER) * bw
    cy = y + bh / 2 + rng.uniform(-BBOX_JITTER, BBOX_JITTER) * bh
    nw = bw * rng.uniform(1 - BBOX_JITTER, 1 + BBOX_JITTER)
    nh = bh * rng.uniform(1 - BBOX_JITTER, 1 + BBOX_JITTER)
    x0 = min(max(int(round(cx - nw / 2)), 0), w - 1)
    y0 = min(max(int(round(cy - nh / 2)), 0), h - 1)
    return image, (x0, y0, max(1, min(int(round(nw)), w - x0)), max(1, min(int(round(nh)), h - y0)))


def augment_crop(crop: np.ndarray, rng=random) -> np.ndarray:
    """Orientation, colour and focus changes. Lesions have no canonical orientation, and endoscope
    processors, light sources and capsules render colour very differently."""
    if rng.random() < 0.5:
        crop = crop[:, ::-1]
    if rng.random() < 0.5:
        crop = crop[::-1]
    crop = np.ascontiguousarray(np.rot90(crop, rng.randint(0, 3)))
    hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV).astype(np.float32)
    hsv[..., 0] = (hsv[..., 0] + rng.uniform(-6, 6)) % 180
    hsv[..., 1] *= rng.uniform(0.85, 1.15)
    crop = cv2.cvtColor(np.clip(hsv, 0, [179, 255, 255]).astype(np.uint8), cv2.COLOR_HSV2BGR).astype(np.float32)
    mean = crop.mean()
    crop = (crop - mean) * rng.uniform(0.8, 1.2) + mean * rng.uniform(0.8, 1.2)
    crop = np.clip(crop, 0, 255).astype(np.uint8)
    if rng.random() < 0.2:
        crop = cv2.GaussianBlur(crop, (0, 0), rng.uniform(0.5, 1.2))
    return crop


class LesionDataset(Dataset):
    """Lesion crops built with the same at_working_size + crop_lesion + to_model_input calls as
    inference, so the model is trained at the resolution and framing it gets in use."""

    def __init__(self, samples: list[Sample], size: int, margin: float, augment: bool,
                 mean=MEAN, std=STD, working_size: Optional[int] = None):
        self.samples, self.size, self.margin, self.augment = samples, size, margin, augment
        self.mean, self.std, self.working_size = mean, std, working_size

    def __len__(self) -> int:
        return len(self.samples)

    def crop(self, i: int) -> np.ndarray:
        s = self.samples[i]
        image = read_image(s.path)
        image, bbox = at_working_size(image, s.bbox_in(image.shape), self.working_size)
        if self.augment:
            # torch reseeds the `random` module in every worker, so workers do not repeat each other.
            image, bbox = jitter_view(image, bbox)
        crop = crop_lesion(image, bbox, self.size, self.margin)
        return augment_crop(crop) if self.augment else crop

    def __getitem__(self, i: int):
        x = to_model_input([self.crop(i)], self.mean, self.std)[0]
        return torch.from_numpy(x), self.samples[i].category


# ----------------------------------------------------------------------------- split


def grouped_split(group_counts: dict[str, np.ndarray], fractions: dict[str, float], seed: int,
                  base_train: Optional[np.ndarray] = None) -> dict[str, str]:
    """Assign whole groups (patients) to splits, keeping each split's class mix close to the overall one.

    Groups carrying a large share of a rare class are placed first, each into the split
    furthest below its target for that group's classes. A final pass moves a group so
    that val and test contain every class where the data allow it. `base_train` holds the
    class counts already fixed in the training split (patients an --init checkpoint was
    trained on); the targets include them, so new patients go mostly to val and test.
    """
    names = sorted(group_counts)
    random.Random(seed).shuffle(names)
    counts = np.stack([group_counts[g] for g in names]).astype(np.float64)
    base = np.zeros(counts.shape[1]) if base_train is None else np.asarray(base_train, np.float64)
    total = counts.sum(axis=0) + base
    frac = np.array([fractions[s] for s in SPLITS])
    target = frac[:, None] * total[None, :]
    current = np.zeros_like(target)
    current[SPLITS.index("train")] = base
    share = counts / np.maximum(total, 1)
    order = sorted(range(len(names)), key=lambda i: -share[i].max())
    assigned: dict[int, int] = {}
    for i in order:
        need = ((target - current) / np.maximum(total, 1)) @ counts[i]
        room = target.sum(axis=1) - current.sum(axis=1)
        s = max(range(len(SPLITS)), key=lambda j: (round(float(need[j]), 12), room[j]))
        assigned[i] = s
        current[s] += counts[i]

    def carriers(split_idx: int, k: int) -> list[int]:
        return [i for i, s in assigned.items() if s == split_idx and counts[i, k] > 0]

    for fix in (SPLITS.index("test"), SPLITS.index("val")):
        for k in range(counts.shape[1]):
            if total[k] == 0 or carriers(fix, k):
                continue
            for donor in (SPLITS.index("train"), *(j for j in range(len(SPLITS)) if j not in (fix, 0))):
                pool = carriers(donor, k)
                if len(pool) >= 2:
                    assigned[min(pool, key=lambda i: counts[i].sum())] = fix
                    break
    return {names[i]: SPLITS[s] for i, s in assigned.items()}


def group_hash(salt: str, group: str) -> str:
    """Salted, truncated SHA-256 of a patient (or image) group, so model files can record who
    was in the development data without carrying the pseudonymous IDs themselves."""
    return hashlib.sha256(f"{salt}:{group}".encode()).hexdigest()[:16]


def split_samples(samples: list[Sample], val_frac: float, test_frac: float, seed: int,
                  pinned_train: set[str] | frozenset[str] = frozenset(),
                  ) -> tuple[dict[str, list[Sample]], dict, list[str]]:
    """Patient-grouped split. Groups in `pinned_train` (patients an --init checkpoint was
    trained on) always go to train: in val or test they would inflate every metric."""
    warn: list[str] = []
    fixed: dict[str, str] = {}
    conflicts: dict[str, set[str]] = {}
    for s in samples:
        if s.split:
            prev = fixed.setdefault(s.group, s.split)
            if prev != s.split:
                conflicts.setdefault(s.group, {prev}).add(s.split)
    if conflicts:
        listed = "; ".join(f"{g}: {sorted(v)}" for g, v in list(conflicts.items())[:10])
        fail(f"The split column puts {_n(len(conflicts), 'patient')} in more than one split ({listed}). "
             "Frames of one patient on both sides of a split inflate results.")

    leaked = sorted(g for g in pinned_train if fixed.get(g, "train") != "train")
    if leaked:
        fail(f"The split column puts {_n(len(leaked), 'patient')} the --init checkpoint was trained on in val "
             f"or test ({', '.join(leaked[:10])}). The fine-tuned model would be scored on patients it has "
             "already fitted; put them in train or start without --init.")

    auto: dict[str, np.ndarray] = {}
    base = np.zeros(len(CATEGORIES), np.int64)
    pinned: dict[str, str] = {}
    for s in samples:
        if s.group in fixed:
            continue
        if s.group in pinned_train:
            pinned[s.group] = "train"
            base[s.category] += 1
            continue
        auto.setdefault(s.group, np.zeros(len(CATEGORIES), np.int64))[s.category] += 1
    fractions = {"train": 1.0 - val_frac - test_frac, "val": val_frac, "test": test_frac}
    if auto:
        if len(auto) < 3 and not fixed and not pinned:
            fail(f"Only {_n(len(auto), 'patient')} to split; at least 3 are needed (one per split).")
        group_split = {**fixed, **pinned, **grouped_split(auto, fractions, seed, base if pinned else None)}
    else:
        group_split = {**fixed, **pinned}
    method = ("explicit split column" if not auto and not pinned else
              "patient-grouped automatic" if not fixed else "explicit split column + patient-grouped automatic")
    if pinned:
        method += ", with the --init checkpoint's training patients kept in train"

    parts = {sp: [s for s in samples if group_split[s.group] == sp] for sp in SPLITS}
    for sp in SPLITS:
        if not parts[sp]:
            if pinned and not auto:
                fail(f"The {sp} split is empty: every patient was used to train the --init checkpoint. "
                     "Add patients it has not seen, for validation and testing.")
            fail(f"The {sp} split is empty. Add rows with split={sp} or remove the split column.")
    groups = {sp: {s.group for s in parts[sp]} for sp in SPLITS}
    for a, b in (("train", "val"), ("train", "test"), ("val", "test")):
        overlap = groups[a] & groups[b]
        if overlap:
            raise AssertionError(f"patients in both {a} and {b}: {sorted(overlap)[:10]}")

    no_patient = sum(1 for s in samples if not s.patient_id)
    if no_patient:
        warn.append(
            f"{no_patient} of {len(samples)} rows have no patient_id, so they were split image by image. "
            "Frames of the same lesion or patient can then land in both training and test, which inflates "
            "every metric reported here. Add a patient_id column before trusting these numbers."
        )
    for sp in ("val", "test"):
        absent = [c for k, c in enumerate(CATEGORIES) if not any(s.category == k for s in parts[sp])]
        if absent:
            warn.append(f"The {sp} split has no {', '.join(absent)} examples (too few patients with them).")

    info = {
        "method": method,
        "unit": ("patient" if not no_patient else "image (no patient_id)" if no_patient == len(samples)
                 else "patient where given, otherwise image"),
        "seed": seed,
        "fractions": fractions,
        "rows_without_patient_id": no_patient,
        "patients_kept_in_train_from_init": len(pinned),
        "sizes": {sp: len(parts[sp]) for sp in SPLITS},
        "patients": {sp: len(groups[sp]) for sp in SPLITS},
        "class_counts": {sp: class_counts(parts[sp]) for sp in SPLITS},
        "images": {sp: [str(Path(s.dataset) / s.image) for s in parts[sp]] for sp in SPLITS},
        "patient_ids": {sp: sorted({s.patient_id for s in parts[sp] if s.patient_id}) for sp in SPLITS},
    }
    return parts, info, warn


def class_counts(samples: list[Sample]) -> dict[str, int]:
    counts = np.bincount([s.category for s in samples], minlength=len(CATEGORIES))
    return {c: int(n) for c, n in zip(CATEGORIES, counts)}


def print_class_counts(parts: dict[str, list[Sample]]) -> None:
    print("Class counts".ljust(14) + "".join(c.rjust(14) for c in CATEGORIES) + "total".rjust(9) + "patients".rjust(10))
    for sp in SPLITS:
        counts = class_counts(parts[sp])
        print(f"  {sp:<12}" + "".join(str(counts[c]).rjust(14) for c in CATEGORIES)
              + str(len(parts[sp])).rjust(9) + str(len({s.group for s in parts[sp]})).rjust(10))


# ----------------------------------------------------------------------------- models


def inverse_frequency_weights(counts: np.ndarray) -> np.ndarray:
    counts = np.asarray(counts, np.float64)
    return counts.sum() / (len(counts) * counts)


def logit_adjustment(class_weights: np.ndarray) -> np.ndarray:
    """Per-class offset that undoes the prior shift of a class-weighted loss.

    Weighted cross-entropy fits q(k|x) proportional to w_k p(k|x), so its logits carry an extra
    log w_k. Subtracting it restores the training-set class prior; a single temperature cannot,
    and without it rare classes (cancer) get inflated probabilities. Centred, as softmax ignores
    a constant.
    """
    offset = -np.log(np.asarray(class_weights, np.float64))
    return offset - offset.mean()


def make_criterion(class_weights: np.ndarray, device) -> nn.Module:
    return nn.CrossEntropyLoss(weight=torch.tensor(class_weights, dtype=torch.float32, device=device),
                               label_smoothing=LABEL_SMOOTHING)


class LogitAdjusted(nn.Module):
    """The trained network plus the logit_adjustment offset, which is what gets exported."""

    def __init__(self, model: nn.Module, offset: np.ndarray):
        super().__init__()
        self.model = model
        self.register_buffer("offset", torch.as_tensor(np.asarray(offset), dtype=torch.float32))

    def forward(self, x):
        return self.model(x) + self.offset


class TinyNet(nn.Module):
    """Small CNN (about 100k parameters) for tests and smoke runs. Use a real backbone for real data."""

    def __init__(self, n_classes: int = len(CATEGORIES)):
        super().__init__()
        chans = [3, 16, 32, 64, 128]
        layers: list[nn.Module] = []
        for cin, cout in zip(chans, chans[1:]):
            layers += [nn.Conv2d(cin, cout, 3, padding=1, bias=False), nn.BatchNorm2d(cout),
                       nn.ReLU(inplace=True), nn.MaxPool2d(2)]
        self.features = nn.Sequential(*layers)
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.head = nn.Sequential(nn.Flatten(), nn.Dropout(0.2), nn.Linear(chans[-1], n_classes))

    def forward(self, x):
        return self.head(self.pool(self.features(x)))


def build_model(backbone: str, pretrained: bool) -> nn.Module:
    n = len(CATEGORIES)
    if backbone == "tiny":
        if pretrained:
            fail("--pretrained is not available for the tiny backbone.")
        return TinyNet(n)
    import torchvision

    if pretrained:
        weights = torchvision.models.get_model_weights(backbone).DEFAULT
        try:
            model = torchvision.models.get_model(backbone, weights=weights)
        except Exception as e:  # network errors surface as URLError, OSError or RuntimeError
            fail(f"Could not obtain ImageNet weights for {backbone} ({type(e).__name__}: {e}).\n"
                 f"This machine may be offline. On a connected machine download\n  {weights.url}\n"
                 "copy it here and pass --weights PATH instead of --pretrained.")
    else:
        model = torchvision.models.get_model(backbone, weights=None)
    if backbone.startswith("resnet"):
        model.fc = nn.Linear(model.fc.in_features, n)
    else:  # efficientnet_b0, convnext_tiny: the last classifier layer is the linear head
        model.classifier[-1] = nn.Linear(model.classifier[-1].in_features, n)
    return model


def _unwrap_state_dict(obj) -> dict[str, torch.Tensor]:
    for key in ("state_dict", "model", "model_state_dict"):
        if isinstance(obj, dict) and isinstance(obj.get(key), dict):
            obj = obj[key]
            break
    if not isinstance(obj, dict):
        fail("The weights file does not contain a state_dict.")
    return {k.removeprefix("module."): v for k, v in obj.items() if isinstance(v, torch.Tensor)}


def _short(keys: list[str], n: int = 6) -> str:
    return ", ".join(keys[:n]) + (f" ... (+{len(keys) - n} more)" if len(keys) > n else "")


def load_backbone_weights(model: nn.Module, path: Path, backbone: str) -> None:
    """Load a local backbone state_dict (e.g. torchvision ImageNet weights) non-strictly.

    Tensors whose shape does not match (the 1000-class ImageNet head) are skipped:
    the 3-class head is always trained from scratch.
    """
    if not path.is_file():
        fail(f"--weights file {path} not found.")
    state = _unwrap_state_dict(torch.load(path, map_location="cpu", weights_only=True))
    own = model.state_dict()
    usable = {k: v for k, v in state.items() if k in own and own[k].shape == v.shape}
    mismatched = sorted(k for k in state if k in own and own[k].shape != state[k].shape)
    unexpected = sorted(k for k in state if k not in own)
    missing = sorted(model.load_state_dict(usable, strict=False).missing_keys)
    if not usable:
        fail(f"No tensor in {path} matches the {backbone} backbone. Is it a state_dict for {backbone}?")
    print(f"Loaded {len(usable)}/{len(own)} tensors from {path}")
    if mismatched:
        print(f"  skipped, shape mismatch: {_short(mismatched)}")
    if unexpected:
        print(f"  unexpected keys, ignored: {_short(unexpected)}")
    if missing:
        print(f"  missing keys, left at random initialisation: {_short(missing)}")
    body_missing = [k for k in missing if not k.startswith(HEAD_PREFIX[backbone])]
    if body_missing:
        print(f"WARNING: {len(body_missing)} backbone tensors were not in {path}; check it matches {backbone}.")


def load_checkpoint(path: Path) -> tuple[dict[str, torch.Tensor], dict]:
    if not path.is_file():
        fail(f"--init checkpoint {path} not found.")
    ckpt = torch.load(path, map_location="cpu", weights_only=True)
    config = ckpt.get("config", {}) if isinstance(ckpt, dict) else {}
    return _unwrap_state_dict(ckpt), config


# ----------------------------------------------------------------------------- training


def softmax(logits: np.ndarray, temperature: float = 1.0) -> np.ndarray:
    z = np.asarray(logits, np.float64) / temperature
    z -= z.max(axis=1, keepdims=True)
    e = np.exp(z)
    return e / e.sum(axis=1, keepdims=True)


def nll(logits: np.ndarray, y: np.ndarray, temperature: float = 1.0) -> float:
    z = np.asarray(logits, np.float64) / temperature
    z -= z.max(axis=1, keepdims=True)
    log_p = z - np.log(np.exp(z).sum(axis=1, keepdims=True))
    return max(0.0, float(-log_p[np.arange(len(y)), y].mean()))


def fit_temperature(logits: np.ndarray, y: np.ndarray) -> float:
    """Temperature minimising validation NLL, searched over log T (grid, then golden section).

    NLL is convex in 1/T, hence unimodal in log T, so the bracketed search finds the optimum.
    """
    lo, hi = (math.log(t) for t in TEMPERATURE_RANGE)
    grid = np.linspace(lo, hi, 121)
    values = [nll(logits, y, math.exp(t)) for t in grid]
    i = int(np.argmin(values))
    a, b = grid[max(i - 1, 0)], grid[min(i + 1, len(grid) - 1)]
    g = (math.sqrt(5) - 1) / 2
    for _ in range(60):
        c, d = b - g * (b - a), a + g * (b - a)
        if nll(logits, y, math.exp(c)) < nll(logits, y, math.exp(d)):
            b = d
        else:
            a = c
    return float(np.clip(math.exp((a + b) / 2), *TEMPERATURE_RANGE))


def selection_score(report: dict) -> tuple[str, float]:
    """Early-stopping metric: macro AUROC, or balanced accuracy when no class AUROC is defined."""
    if report["macro_auroc"] is not None:
        return "macro_auroc", float(report["macro_auroc"])
    return "balanced_accuracy", float(report["balanced_accuracy"] or 0.0)


def run_epoch(model, loader, criterion, opt, sched, scaler, device, amp: bool) -> float:
    model.train()
    total, n = 0.0, 0
    for x, y in loader:
        x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
        with torch.autocast(device_type="cuda", dtype=torch.float16) if amp else contextlib.nullcontext():
            loss = criterion(model(x), y)
        opt.zero_grad(set_to_none=True)
        scaler.scale(loss).backward()
        scaler.unscale_(opt)
        nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP)
        scaler.step(opt)
        scaler.update()
        sched.step()
        total += float(loss.item()) * len(y)
        n += len(y)
    return total / max(n, 1)


@torch.no_grad()
def torch_logits(model, loader, device) -> tuple[np.ndarray, np.ndarray]:
    model.eval()
    logits, labels = [], []
    for x, y in loader:
        logits.append(model(x.to(device)).float().cpu().numpy())
        labels.append(y.numpy())
    return np.concatenate(logits), np.concatenate(labels)


def cosine_with_warmup(warmup_steps: int, total_steps: int):
    def factor(step: int) -> float:
        if step < warmup_steps:
            return (step + 1) / warmup_steps
        progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
        return 0.5 * (1 + math.cos(math.pi * min(1.0, progress)))
    return factor


# ----------------------------------------------------------------------------- ONNX


def export_onnx(model: nn.Module, size: int, path: Path) -> int:
    """Export with input "image" and output "logits" and a dynamic batch axis; return the opset used.

    The TorchScript exporter gives exact opset 17. Newer torch releases may drop it, in which case
    the torch.export-based exporter is used, which needs opset 18 or later.
    """
    import onnx

    model = model.cpu().eval()
    dummy = torch.zeros(2, 3, size, size)
    common = dict(input_names=["image"], output_names=["logits"],
                  dynamic_axes={"image": {0: "batch"}, "logits": {0: "batch"}})
    has_dynamo = "dynamo" in inspect.signature(torch.onnx.export).parameters
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        try:
            torch.onnx.export(model, (dummy,), str(path), opset_version=17,
                              **({"dynamo": False} if has_dynamo else {}), **common)
        except Exception as e:
            if not has_dynamo:
                raise
            print(f"TorchScript ONNX exporter unavailable ({type(e).__name__}); using the torch.export exporter.")
            torch.onnx.export(model, (dummy,), str(path), opset_version=18, dynamo=True,
                              external_data=False, **common)
    proto = onnx.load(str(path))
    onnx.checker.check_model(proto)
    return next(o.version for o in proto.opset_import if o.domain in ("", "ai.onnx"))


def onnx_session(path: Path):
    try:
        import onnxruntime as ort
    except ImportError:
        fail("onnxruntime is needed to verify and evaluate the exported model: pip install onnxruntime")
    return ort.InferenceSession(str(path), providers=["CPUExecutionProvider"])


def onnx_logits(session, dataset: LesionDataset, batch: int) -> np.ndarray:
    return batched_logits(lambda x: session.run(["logits"], {"image": x})[0], dataset, batch)


def batched_logits(run, dataset: LesionDataset, batch: int) -> np.ndarray:
    """`run` maps a to_model_input batch to (N, 3) logits."""
    out = []
    for start in range(0, len(dataset), batch):
        crops = [dataset.crop(i) for i in range(start, min(start + batch, len(dataset)))]
        out.append(run(to_model_input(crops, dataset.mean, dataset.std)))
    return np.concatenate(out).astype(np.float64) if out else np.zeros((0, len(CATEGORIES)))


def check_onnx_parity(model: nn.Module, session, dataset: LesionDataset, size: int) -> float:
    """Compare ONNX Runtime with PyTorch on real crops and on several batch sizes."""
    model = model.cpu().eval()
    real = to_model_input([dataset.crop(i) for i in range(min(8, len(dataset)))], dataset.mean, dataset.std)
    rng = np.random.default_rng(0)
    batches = [real, real[:1], rng.normal(size=(3, 3, size, size)).astype(np.float32)]
    worst = 0.0
    with torch.no_grad():
        for x in batches:
            ref = model(torch.from_numpy(x)).numpy()
            got = session.run(["logits"], {"image": x})[0]
            if got.shape != ref.shape:
                fail(f"ONNX output shape {got.shape} differs from PyTorch {ref.shape}.")
            worst = max(worst, float(np.abs(got - ref).max()))
    if worst > ONNX_ATOL:
        fail(f"ONNX logits differ from PyTorch by up to {worst:.2e} (tolerance {ONNX_ATOL}). Export is not trustworthy.")
    return worst


# ----------------------------------------------------------------------------- reporting


def _clean(obj):
    """Replace NaN/inf with None so outputs are strict JSON."""
    if isinstance(obj, float):
        return obj if math.isfinite(obj) else None
    if isinstance(obj, dict):
        return {k: _clean(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_clean(v) for v in obj]
    if isinstance(obj, (np.floating, np.integer)):
        return _clean(obj.item())
    return obj


def write_json(path: Path, obj) -> None:
    path.write_text(json.dumps(_clean(obj), indent=2, allow_nan=False))


def write_predictions(path: Path, samples: list[Sample], probs: np.ndarray, abstain_below: float) -> None:
    """`predicted` is the argmax; `ai_category` is what the report would show (the deployed decision)."""
    decisions = decide(probs, abstain_below)
    with path.open("w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["image", "patient_id", "true_category", "predicted",
                    *[f"p_{c}" for c in CATEGORIES], "abstained", "ai_category", "label", "dataset"])
        for s, p, d in zip(samples, probs, decisions):
            w.writerow([s.image, s.patient_id, CATEGORIES[s.category], CATEGORIES[int(np.argmax(p))],
                        *[f"{v:.4f}" for v in p], bool(d == ABSTAIN),
                        "indeterminate" if d == ABSTAIN else CATEGORIES[int(d)], s.label, s.dataset])


def _pct(v: Optional[float]) -> str:
    return "n/a" if v is None else f"{100 * v:.1f}%"


def _num(v: Optional[float]) -> str:
    return "n/a" if v is None else f"{v:.3f}"


def _ci(ci: Optional[dict], metric: str, fmt=_num) -> str:
    m = (ci or {}).get(metric) or {}
    if m.get("low") is not None:
        exact = f", exact: {m['note']}" if m.get("method") == "exact" else ""
        return f" [95% CI {fmt(m['low'])}-{fmt(m['high'])}{exact}]"
    return f" [95% CI {m['note']}]" if m.get("note") else ""


def print_summary(title: str, report: dict, abstain_below: float, ci: Optional[dict] = None) -> list[str]:
    """Print the clinically relevant numbers; return warnings about unreliable estimates."""
    print(f"\n== {title} ({report['n']} images) ==")
    print("Confusion matrix (rows = histology, columns = AI prediction)")
    print(" " * 16 + "".join(c.rjust(14) for c in CATEGORIES))
    for c, row in zip(CATEGORIES, report["confusion_matrix"]):
        print(f"  {c:<14}" + "".join(str(v).rjust(14) for v in row))
    print(f"\n  {'class':<14}{'n':>6}{'sens':>9}{'spec':>9}{'PPV':>9}{'NPV':>9}{'AUROC':>9}")
    for c in CATEGORIES:
        m = report["per_class"][c]
        print(f"  {c:<14}{m['support']:>6}{_pct(m['sensitivity']):>9}{_pct(m['specificity']):>9}"
              f"{_pct(m['ppv']):>9}{_pct(m['npv']):>9}{_num(m['auroc']):>9}")
    print(f"\n  Accuracy {_pct(report['accuracy'])}{_ci(ci, 'accuracy', _pct)}; balanced accuracy "
          f"{_pct(report['balanced_accuracy'])}{_ci(ci, 'balanced_accuracy', _pct)}")
    print(f"  Macro AUROC {_num(report['macro_auroc'])}{_ci(ci, 'macro_auroc')} "
          f"(over {report['macro_auroc_n_classes']} classes)")
    neo, cancer, sel = report["neoplastic"], report["cancer"], report["selective"]
    if sel:
        print(f"\n  Deployed decision, as the report shows it (abstain below {abstain_below:.2f}; benign only when "
              "P(benign) > P(precancerous) + P(cancerous)):")
        print(f"    answered {sel['n_covered']}/{sel['n']} ({_pct(sel['coverage'])}), "
              f"accuracy on answered lesions {_pct(sel['accuracy_covered'])}")
        dn, dc = sel.get("neoplastic"), sel.get("cancer")
        if dn:
            print(f"    Neoplastic vs benign on answered lesions: sensitivity {_pct(dn['sensitivity'])}, "
                  f"specificity {_pct(dn['specificity'])}, PPV {_pct(dn['ppv'])}, NPV {_pct(dn['npv'])}"
                  f"{_ci(ci, 'neoplastic_npv', _pct)}; {dn['neoplastic_abstained']} neoplastic lesions given "
                  "no category (not called benign)")
            print(f"    {PIVI_NOTE}")
        if dc:
            print(f"    Cancers ({dc['n_positive']}): called cancer {dc['called_cancerous']}, precancerous "
                  f"{dc['called_precancerous']}, benign {dc['called_benign']}, no category {dc['abstained']}; "
                  f"sensitivity {_pct(dc['sensitivity'])}")
    if neo:
        print(f"  All lesions, including those the model abstains on: neoplastic (P >= {neo['threshold']}) "
              f"sensitivity {_pct(neo['sensitivity'])}, specificity {_pct(neo['specificity'])}, "
              f"NPV {_pct(neo['npv'])}; cancer (argmax) sensitivity {_pct(cancer['sensitivity'])} "
              f"({cancer['tp']}/{cancer['n_positive']}), specificity {_pct(cancer['specificity'])}")
    print(f"  Calibration error (ECE, {report['ece_bins']} bins): {_num(report['ece'])}")
    low = [f"{c} (n={n})" for c, n in report["class_counts"].items() if n < MIN_RELIABLE_PER_CLASS]
    warn = []
    if low:
        warn.append(f"{title}: fewer than {MIN_RELIABLE_PER_CLASS} examples of {', '.join(low)}. "
                    "These estimates are unreliable; confidence intervals are wide.")
        print(f"\nWARNING: {warn[-1]}")
    return warn


# ----------------------------------------------------------------------------- modes


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def dataset_fingerprint(root: Path, rows: int) -> dict:
    """Identifies a --data directory independently of where it is stored or how its path was typed."""
    synthetic = None
    info = root / "dataset.json"
    if info.is_file():
        try:
            synthetic = json.loads(info.read_text()).get("synthetic")
        except (ValueError, AttributeError):
            synthetic = None
    return {"path": str(root.resolve()), "labels_sha256": file_sha256(root / "labels.csv"), "rows": rows,
            "synthetic": synthetic if isinstance(synthetic, bool) else None}


def _dataset_list(value) -> list[dict]:
    return [d for d in value or [] if isinstance(d, dict)]


def init_lineage(init_config: dict) -> list[dict]:
    """Data sets an --init checkpoint and its own ancestors were trained on (path, labels.csv hash)."""
    out = _dataset_list(init_config.get("init_training_datasets"))
    own = init_config.get("training_datasets")
    if own is None:  # checkpoints of earlier versions record paths only
        own = [{"path": str(p)} for p in init_config.get("training_data") or []]
    seen, unique = set(), []
    for d in out + _dataset_list(own):
        key = (d.get("path"), d.get("labels_sha256"))
        if key not in seen:
            seen.add(key)
            unique.append(d)
    return unique


def development_manifest_path(model_path: Path) -> Path:
    return model_path.with_suffix(DEVELOPMENT_SUFFIX)


def load_development_manifest(model_path: Path, meta: dict, model_sha256: str) -> tuple[Optional[dict], str]:
    """Salted hashes of the development data (patients and images) and where they came from."""
    path = development_manifest_path(model_path)
    if path.is_file():
        try:
            manifest = json.loads(path.read_text())
        except ValueError as e:
            return None, f"{path} is not valid JSON ({e})"
        if manifest.get("model_sha256") not in (None, model_sha256):
            return None, f"{path} belongs to another model file (SHA-256 differs)"
        if manifest.get("salt") and isinstance(manifest.get("sha256_16"), list):
            return manifest, str(path)
        return None, f"{path} holds no development hashes"
    legacy = meta.get("development_patients")  # sidecars of earlier versions carried the hashes
    if isinstance(legacy, dict) and legacy.get("salt") and isinstance(legacy.get("sha256_16"), list):
        return legacy, f"{model_path.with_suffix('.json')} (development_patients)"
    return None, f"no {path.name} next to the model"


def development_overlap(meta: dict, data_dirs: list[Path], samples: list[Sample],
                        manifest: Optional[dict] = None) -> dict:
    """What the evaluation data share with the model's development (train/val/test) data, its --init
    checkpoint's included: the same directory, a labels.csv with identical content (a copy),
    patients (IDs compared ignoring case) or byte-identical image files. Absence of overlap does not
    make a data set external; only its source (another centre, device or period) can."""
    trained = meta.get("training_data") or []
    lineage = _dataset_list(meta.get("training_datasets")) + _dataset_list(meta.get("init_training_datasets"))
    trained_paths = {str(Path(p).resolve()) for p in ([trained] if isinstance(trained, str) else trained)}
    trained_paths |= {str(Path(d["path"]).resolve()) for d in lineage if d.get("path")}
    fingerprints = {d.get("labels_sha256") for d in lineage if d.get("labels_sha256")}
    without_id = sum(1 for s in samples if not s.patient_id)
    out: dict = {
        "same_directory": [str(d) for d in data_dirs if str(d.resolve()) in trained_paths],
        "same_labels_csv": [str(d) for d in data_dirs if file_sha256(d / "labels.csv") in fingerprints],
        "shared_patients": None,
        "shared_patient_examples": [],
        "shared_images": None,
        "patients_checked": False,
        "images_checked": False,
        "rows_without_patient_id": without_id,
    }
    if manifest:
        salt, known = manifest["salt"], set(manifest["sha256_16"])
        shared = sorted({s.patient_id for s in samples if s.patient_id and any(
            group_hash(salt, k) in known for k in (patient_key(s.patient_id), s.patient_id))})
        out["images_checked"] = all(s.image_sha256 for s in samples)
        out["shared_images"] = sum(1 for s in samples if s.image_sha256
                                   and group_hash(salt, f"image:{s.image_sha256}") in known)
        if without_id < len(samples):  # patients can only be compared where IDs are given
            out.update(shared_patients=len(shared), shared_patient_examples=shared[:5], patients_checked=True)
    out["overlap"] = bool(out["same_directory"] or out["same_labels_csv"] or out["shared_patients"]
                          or out["shared_images"])
    return out


def train(args) -> None:
    if args.pretrained and args.weights:
        fail("Use either --pretrained (download) or --weights PATH (local file), not both.")
    if not 0 < args.val_frac < 1 or not 0 < args.test_frac < 1 or args.val_frac + args.test_frac >= 1:
        fail("--val-frac and --test-frac must be in (0, 1) and sum to less than 1.")
    seed_everything(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    amp = device.type == "cuda"
    args.out.mkdir(parents=True, exist_ok=True)
    name = args.name or f"lesion-cadx-{args.backbone}"
    # A distinct default version per run, so the audit log can tell models apart.
    version = args.version or datetime.now(timezone.utc).strftime("%Y.%m.%d-%H%M%S")
    working_size = get_config(args.modality).working_size
    warn: list[str] = []

    samples, notes = load_samples(args.data)
    for note in notes:
        print(f"Note: {note}")

    init_state, init_config = load_checkpoint(args.init) if args.init else (None, {})
    salt = str(init_config.get("group_salt") or secrets.token_hex(8))
    inherited = set(init_config.get("trained_group_hashes") or [])
    if args.init and "trained_group_hashes" not in init_config:
        warn.append(f"{args.init} does not record which patients it was trained on, so some of them may now be "
                    "in val or test, which inflates every metric. Retrain the checkpoint with this version.")
        print(f"WARNING: {warn[-1]}")
    # A patient is pinned when its ID or any of its images was in the checkpoint's training data.
    pinned = {s.group for s in samples if any(group_hash(salt, k) in inherited for k in lookup_keys(s))}
    parts, split_info, split_warn = split_samples(samples, args.val_frac, args.test_frac, args.seed, pinned)
    write_json(args.out / "split.json", split_info)
    print(f"Split: {split_info['method']}, unit = {split_info['unit']} (written to {args.out / 'split.json'})")
    if pinned:
        images = sum(g.startswith("image:") for g in pinned)
        kept = " and ".join(t for t in (_n(len(pinned) - images, "patient") if len(pinned) > images else "",
                                         _n(images, "image") + " without patient_id" if images else "") if t)
        print(f"  {kept} the --init checkpoint was trained on kept in the training split.")
    elif inherited:
        print(f"Note: none of the patients or images {args.init} was trained on are in --data. That is expected "
              "for new data; if --data includes the checkpoint's own data, its images have changed (re-encoded "
              "or edited) and they cannot be kept out of val and test.")
    print_class_counts(parts)
    for w in split_warn:
        print(f"WARNING: {w}")
    warn += split_warn

    counts = np.bincount([s.category for s in parts["train"]], minlength=len(CATEGORIES))
    if (counts == 0).any():
        absent = [c for c, n in zip(CATEGORIES, counts) if n == 0]
        fail(f"The training split has no {', '.join(absent)} examples; the model could never predict them.")
    class_weights = inverse_frequency_weights(counts)
    adjust = logit_adjustment(class_weights)
    print("Class weights (inverse frequency): "
          + ", ".join(f"{c} {w:.2f}" for c, w in zip(CATEGORIES, class_weights))
          + "; their prior shift is removed from the exported logits")

    model = build_model(args.backbone, args.pretrained and not args.init)
    if args.weights:
        load_backbone_weights(model, args.weights, args.backbone)
    if args.init:
        if init_config.get("backbone") not in (None, args.backbone):
            fail(f"--init checkpoint is a {init_config['backbone']} model but --backbone is {args.backbone}.")
        try:
            model.load_state_dict(init_state)
        except RuntimeError as e:
            fail(f"--init checkpoint does not fit the {args.backbone} model: {e}")
        print(f"Fine-tuning from {args.init} (trained on {init_config.get('training_data', 'unknown data')})")
        for key, value in (("input_size", args.size), ("crop_margin", args.crop_margin),
                           ("modality", args.modality), ("working_size", working_size)):
            if key in init_config and init_config[key] != value:
                print(f"Note: --init model used {key}={init_config[key]}, this run uses {value}.")
    elif args.backbone != "tiny" and not (args.pretrained or args.weights):
        print(f"WARNING: training {args.backbone} from random initialisation. ImageNet weights "
              "(--pretrained or --weights) usually help a lot on small medical data sets.")
    model.to(device)
    print(f"Model {args.backbone}: {sum(p.numel() for p in model.parameters()):,} parameters, device {device}")
    print(f"Images are shrunk to the {args.modality} working size ({working_size} px) before cropping, "
          "as in analysis.")

    gen = torch.Generator().manual_seed(args.seed)
    ds_kw = dict(working_size=working_size)
    train_ds = LesionDataset(parts["train"], args.size, args.crop_margin, augment=True, **ds_kw)
    val_ds = LesionDataset(parts["val"], args.size, args.crop_margin, augment=False, **ds_kw)
    test_ds = LesionDataset(parts["test"], args.size, args.crop_margin, augment=False, **ds_kw)
    loader_kw = dict(num_workers=args.workers, pin_memory=amp, persistent_workers=args.workers > 0)
    # Drop the last batch only when it would hold a single image, which BatchNorm cannot train on.
    train_loader = DataLoader(train_ds, args.batch, shuffle=True, drop_last=len(train_ds) % args.batch == 1,
                              generator=gen, **loader_kw)
    val_loader = DataLoader(val_ds, args.batch, shuffle=False, **loader_kw)

    criterion = make_criterion(class_weights, device)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    steps = len(train_loader)
    sched = torch.optim.lr_scheduler.LambdaLR(opt, cosine_with_warmup(steps, steps * args.epochs))
    scaler = torch.amp.GradScaler(device.type, enabled=amp)

    datasets = [dataset_fingerprint(d, sum(1 for s in samples if s.dataset == str(d))) for d in args.data]
    lineage = init_lineage(init_config) if args.init else []
    synthetic_only = all(d["synthetic"] is True for d in datasets) and init_config.get("synthetic_only", True) is True
    trained_hashes = sorted(inherited | {group_hash(salt, k) for s in parts["train"] for k in identity_keys(s)})
    config = {
        "name": name, "version": version, "backbone": args.backbone, "classes": CATEGORIES,
        "input_size": args.size, "mean": MEAN, "std": STD, "crop_margin": args.crop_margin,
        "modality": args.modality, "working_size": working_size,
        "training_data": [d["path"] for d in datasets], "training_datasets": datasets,
        "init_training_datasets": lineage, "seed": args.seed,
        "pretrained": "imagenet" if args.pretrained and not args.init else (str(args.weights) if args.weights else None),
        "init": str(args.init.resolve()) if args.init else None,
        "synthetic_only": synthetic_only,
        # Every patient and image this weight lineage has been trained on, for --init of the next run.
        "group_salt": salt, "trained_group_hashes": trained_hashes,
    }
    history: list[dict] = []
    best: Optional[tuple[float, float]] = None
    stale = 0
    for epoch in range(1, args.epochs + 1):
        t0 = time.time()
        loss = run_epoch(model, train_loader, criterion, opt, sched, scaler, device, amp)
        if not math.isfinite(loss):
            fail("Training loss is not finite; lower --lr.")
        logits, y = torch_logits(model, val_loader, device)
        if not np.isfinite(logits).all():
            fail("The model's validation outputs are not finite (training diverged); lower --lr.")
        logits = logits + adjust
        rep = classification_report(y, softmax(logits))
        metric, score = selection_score(rep)
        val_nll = nll(logits, y)
        improved = best is None or score > best[0] + 1e-6 or (abs(score - best[0]) <= 1e-6 and val_nll < best[1])
        history.append({"epoch": epoch, "train_loss": loss, "val_nll": val_nll, "val_accuracy": rep["accuracy"],
                        "val_balanced_accuracy": rep["balanced_accuracy"], "val_macro_auroc": rep["macro_auroc"],
                        "lr": opt.param_groups[0]["lr"], "seconds": time.time() - t0})
        print(f"epoch {epoch}/{args.epochs} loss {loss:.4f} val nll {val_nll:.4f} acc {_pct(rep['accuracy'])} "
              f"bal acc {_pct(rep['balanced_accuracy'])} macro AUROC {_num(rep['macro_auroc'])} "
              f"({time.time() - t0:.1f}s){'  * best' if improved else ''}")
        if improved:
            best, stale = (score, val_nll), 0
            torch.save({"state_dict": {k: v.detach().cpu() for k, v in model.state_dict().items()},
                        "config": {**config, "epoch": epoch, "selection_metric": metric,
                                   "selection_score": score, "val_nll": val_nll}}, args.out / "best.pt")
        else:
            stale += 1
            if stale >= args.patience:
                print(f"Early stopping: no improvement in val {metric} for {args.patience} epochs.")
                break

    state, best_config = load_checkpoint(args.out / "best.pt")
    model = model.cpu()
    model.load_state_dict(state)
    model.eval()
    print(f"\nBest epoch {best_config['epoch']} (val {best_config['selection_metric']} "
          f"{best_config['selection_score']:.4f}). Exporting ONNX...")

    onnx_path = args.out / "model.onnx"
    exported = LogitAdjusted(model, adjust).eval()
    opset = export_onnx(exported, args.size, onnx_path)
    session = onnx_session(onnx_path)
    parity = check_onnx_parity(exported, session, val_ds, args.size)
    print(f"ONNX opset {opset} re-loaded with onnxruntime; max |logit difference| vs PyTorch {parity:.2e}")

    # From here on every number comes from the exported ONNX model, i.e. what will be deployed.
    val_logits, test_logits = onnx_logits(session, val_ds, args.batch), onnx_logits(session, test_ds, args.batch)
    y_val = np.array([s.category for s in parts["val"]])
    y_test = np.array([s.category for s in parts["test"]])
    fitted = temperature = fit_temperature(val_logits, y_val)
    calibrated = True
    val_counts = class_counts(parts["val"])
    absent = [c for c, n in val_counts.items() if n == 0]
    small = [c for c, n in val_counts.items() if 0 < n < MIN_RELIABLE_PER_CLASS]
    if small:
        warn.append(f"The validation set has fewer than {MIN_RELIABLE_PER_CLASS} examples of {', '.join(small)}, "
                    "so early stopping and the fitted temperature (hence probabilities and abstention) are noisy.")
        print(f"WARNING: {warn[-1]}")
    if absent:
        calibrated = False
        warn.append(f"The validation set has no {', '.join(absent)} examples, so the temperature was fitted and "
                    "the best epoch chosen without them: probabilities are not calibrated. Put patients with "
                    f"{', '.join(absent)} lesions in the validation split.")
        print(f"WARNING: {warn[-1]}")
    if len(y_val) and bool((val_logits.argmax(axis=1) == y_val).all()) and fitted < 1.0:
        # Validation NLL then keeps falling as T -> 0, so the fit only finds the search limit.
        # Sharpening on that evidence would push every probability towards 0 or 100%.
        temperature, calibrated = 1.0, False
        warn.append(f"Every validation image is classified correctly, so the temperature cannot be estimated "
                    f"(the fit ran to {fitted:.3f}). Using T = 1, i.e. no sharpening; probabilities are uncalibrated. "
                    "Use a larger or harder validation set.")
        print(f"WARNING: {warn[-1]}")
    elif not TEMPERATURE_RANGE[0] * 1.1 < temperature < TEMPERATURE_RANGE[1] / 1.1:
        calibrated = False
        warn.append(f"Temperature hit its limit ({temperature:.3f}); the validation set is probably too small, "
                    "so probabilities are not calibrated.")
        print(f"WARNING: {warn[-1]}")
    calibration = {
        "temperature": temperature,
        "temperature_fitted": fitted,
        "calibrated": calibrated,
        "class_weights": class_weights.tolist(),
        "logit_adjustment": adjust.tolist(),
        "val_nll_before": nll(val_logits, y_val), "val_nll_after": nll(val_logits, y_val, temperature),
        "val_ece_before": expected_calibration_error(y_val, softmax(val_logits)),
        "val_ece_after": expected_calibration_error(y_val, softmax(val_logits, temperature)),
        "test_ece_before": expected_calibration_error(y_test, softmax(test_logits)),
        "test_ece_after": expected_calibration_error(y_test, softmax(test_logits, temperature)),
    }
    print(f"Temperature scaling on validation logits: T = {temperature:.3f}; "
          f"val ECE {_num(calibration['val_ece_before'])} -> {_num(calibration['val_ece_after'])}, "
          f"val NLL {calibration['val_nll_before']:.4f} -> {calibration['val_nll_after']:.4f}; "
          f"test ECE {_num(calibration['test_ece_before'])} -> {_num(calibration['test_ece_after'])}")

    p_val, p_test = softmax(val_logits, temperature), softmax(test_logits, temperature)
    val_report = classification_report(y_val, p_val, abstain_below=args.abstain_below)
    test_report = classification_report(y_test, p_test, abstain_below=args.abstain_below)
    ci = None
    if args.bootstrap:
        ci = bootstrap_ci(y_test, p_test, [s.group for s in parts["test"]], n=args.bootstrap, seed=args.seed,
                          abstain_below=args.abstain_below)
        test_report["bootstrap_ci"] = ci
    write_predictions(args.out / "predictions_test.csv", parts["test"], p_test, args.abstain_below)
    title = f"Test set (ONNX model, {'calibrated' if calibrated else 'NOT calibrated'})"
    warn += print_summary(title, test_report, args.abstain_below, ci)
    if synthetic_only:
        warn.append("Trained on synthetic images only: a software test, not a clinical model.")

    created = datetime.now(timezone.utc).isoformat(timespec="seconds")
    # Salted hashes of every development patient and image (and of what an --init checkpoint was
    # trained on), so --evaluate-only can detect overlap. Kept out of the sidecar, which travels
    # with the model: short IDs can be recovered from salted hashes by trying them all.
    development = sorted(inherited | {group_hash(salt, k) for sp in SPLITS for s in parts[sp] for k in identity_keys(s)})
    manifest_path = development_manifest_path(onnx_path)
    sidecar = {
        "name": name, "version": version, "task": "lesion-classification", "classes": CATEGORIES,
        "input_size": args.size, "mean": MEAN, "std": STD, "crop_margin": args.crop_margin,
        "modality": args.modality, "working_size": working_size,
        "temperature": temperature, "temperature_fitted": fitted, "calibrated": calibrated,
        "abstain_below": args.abstain_below,
        "min_frame_agreement": args.min_frame_agreement, "output": "logits", "backbone": args.backbone,
        "training_data": config["training_data"], "training_datasets": datasets,
        "init_training_datasets": lineage,
        "synthetic_training_data": synthetic_only,
        "split_sizes": split_info["sizes"],
        "class_weights": class_weights.tolist(), "logit_adjustment": adjust.tolist(),
        "metrics": {"val": val_report, "test": test_report}, "created_utc": created,
        "validation_warnings": warn,
        "intended_use": INTENDED_USE,
        "input_name": "image", "onnx_opset": opset, "split_method": split_info["method"],
        "rows_without_patient_id": split_info["rows_without_patient_id"],
        "pretrained": config["pretrained"], "init": config["init"], "best_epoch": best_config["epoch"],
        "torch_version": torch.__version__,
    }
    write_json(args.out / "model.json", sidecar)
    write_json(manifest_path, {
        "note": "Salted, truncated SHA-256 hashes of the development data's patient IDs and image files, read by "
                "train_classifier.py --evaluate-only. Derived from patient data: keep with the data owner, do not "
                "distribute with the model.",
        "model_sha256": file_sha256(onnx_path), "salt": salt, "sha256_16": development,
    })
    write_json(args.out / "metrics.json", {
        "model": {"name": name, "version": version, "backbone": args.backbone, "onnx": str(onnx_path),
                  "sha256": file_sha256(onnx_path)},
        "created_utc": created,
        "selection": {"metric": best_config["selection_metric"], "best_epoch": best_config["epoch"],
                      "score": best_config["selection_score"]},
        "history": history,
        "calibration": calibration,
        "onnx": {"opset": opset, "max_abs_logit_diff_vs_torch": parity},
        "split_sizes": split_info["sizes"],
        "class_counts": split_info["class_counts"],
        "val": val_report,
        "test": test_report,
        "warnings": warn,
        "intended_use": INTENDED_USE,
    })
    print(f"\nWrote {onnx_path}, model.json, best.pt, {manifest_path.name}, metrics.json, predictions_test.csv, "
          f"split.json to {args.out}")
    print("Distribute model.onnx and model.json only: the other files are derived from patient data.")
    print(f"Model {name} version {version}, SHA-256 {file_sha256(onnx_path)[:12]}")
    print(f"Use it: secondlook analyse VIDEO --modality {args.modality} --classifier {onnx_path}")
    print(INTENDED_USE)


def evaluate_only(args) -> None:
    """Evaluate an exported model on every row of --data, ignoring any split column."""
    model_path: Path = args.evaluate_only
    meta_path = model_path.with_suffix(".json")
    if not model_path.is_file():
        fail(f"{model_path} not found.")
    if not meta_path.is_file():
        fail(f"Sidecar {meta_path} not found; it holds the preprocessing and temperature the model needs.")
    # Load through the runtime classifier, so the sidecar is read and checked, and the class
    # order remapped, exactly as `secondlook analyse --classifier` will do it.
    from secondlook.diagnosis.classifier import OnnxLesionClassifier

    try:
        clf = OnnxLesionClassifier(model_path, providers=["CPUExecutionProvider"], explain=False)
    except (ImportError, ValueError) as e:
        fail(str(e))
    meta = clf.meta
    size, margin = clf.input_size, clf.crop_margin
    mean, std = clf.mean.tolist(), clf.std.tolist()
    temperature, abstain_below = clf.temperature, clf.abstain_below
    args.out.mkdir(parents=True, exist_ok=True)
    warn: list[str] = []

    samples, notes = load_samples(args.data)
    for note in notes:
        print(f"Note: {note}")
    if any(s.split for s in samples):
        print("Note: the split column is ignored; --evaluate-only evaluates every row.")
    manifest, manifest_source = load_development_manifest(model_path, meta, clf.sha256)
    overlap = development_overlap(meta, args.data, samples, manifest)
    overlap["development_manifest"] = manifest_source if manifest else None
    if overlap["same_directory"] or overlap["same_labels_csv"]:
        same = sorted(set(overlap["same_directory"]) | set(overlap["same_labels_csv"]))
        warn.append(f"{', '.join(same)} was used to develop this model or its --init checkpoint (same directory or "
                    "identical labels.csv), so this is NOT an external validation.")
        print(f"WARNING: {warn[-1]}")
    if overlap["shared_patients"]:
        examples = ", ".join(overlap["shared_patient_examples"])
        warn.append(f"{_n(overlap['shared_patients'], 'patient')} in --data (e.g. {examples}) were in this model's "
                    "development data, so this is NOT an external validation.")
        print(f"WARNING: {warn[-1]}")
    if overlap["shared_images"]:
        warn.append(f"{_n(overlap['shared_images'], 'image')} in --data are byte-identical to images in this model's "
                    "development data, so this is NOT an external validation.")
        print(f"WARNING: {warn[-1]}")
    if not manifest:
        print(f"Note: patient and image overlap was not checked ({manifest_source}); only the data directories "
              "and labels.csv files were compared with the model's development data.")
    elif overlap["rows_without_patient_id"]:
        print(f"Note: {overlap['rows_without_patient_id']} of {len(samples)} rows have no patient_id, so patient "
              "overlap could not be checked for them; they were compared by image content only, which does not "
              "recognise a re-encoded, resized or cropped copy.")
    if not overlap["overlap"]:
        checked = ["data directories", "labels.csv files"] + (["image files"] if overlap["images_checked"] else []) \
            + (["patient IDs"] if overlap["patients_checked"] else [])
        print(f"No overlap with the model's development data was detected ({', '.join(checked)} compared). Whether "
              "this is an external validation (another centre, device or period) depends on where the data came from.")
    if not all(s.patient_id for s in samples):
        warn.append("Some rows have no patient_id; confidence intervals treat each such image as its own patient "
                    "and are too narrow if several images show the same lesion.")
        print(f"WARNING: {warn[-1]}")
    if not clf.calibrated:
        warn.append("The model's probabilities are not calibrated (no fitted temperature in its sidecar).")
        print(f"WARNING: {warn[-1]}")

    print(f"Evaluating {meta.get('name', model_path.stem)} {meta.get('version', '')} on {len(samples)} images "
          f"(size {size}, margin {margin}, working size {clf.working_size}, T = {temperature:.3f})")
    dataset = LesionDataset(samples, size, margin, augment=False, mean=mean, std=std, working_size=clf.working_size)
    logits = batched_logits(clf.logits, dataset, args.batch)
    if not np.isfinite(logits).all():
        fail("The model returned non-finite outputs (NaN or infinity) on this data.")
    y = np.array([s.category for s in samples])
    probs = softmax(logits, temperature)
    report = classification_report(y, probs, abstain_below=abstain_below)
    ci = (bootstrap_ci(y, probs, [s.group for s in samples], n=args.bootstrap, seed=args.seed,
                       abstain_below=abstain_below) if args.bootstrap else None)
    if ci:
        report["bootstrap_ci"] = ci
    write_predictions(args.out / "predictions.csv", samples, probs, abstain_below)
    title = f"Evaluation (ONNX model, {'calibrated' if clf.calibrated else 'NOT calibrated'})"
    warn += print_summary(title, report, abstain_below, ci)
    write_json(args.out / "metrics.json", {
        "model": {"name": meta.get("name"), "version": meta.get("version"), "onnx": str(model_path.resolve()),
                  "sha256": clf.sha256, "sidecar_sha256": clf.sidecar_sha256},
        "created_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "data": [str(d.resolve()) for d in args.data],
        "overlap_with_development_data": overlap,
        "note": "Software can detect overlap with the development data but cannot establish that a data set "
                "is external; that depends on its source (another centre, device or period).",
        "temperature": temperature,
        "calibrated": clf.calibrated,
        "abstain_below": abstain_below,
        "ece_uncalibrated": expected_calibration_error(y, softmax(logits)),
        "evaluation": report,
        "warnings": warn,
        "intended_use": meta.get("intended_use", INTENDED_USE),
    })
    print(f"\nWrote metrics.json and predictions.csv to {args.out}")


def _abstain_threshold(value: str) -> float:
    v = float(value)
    if not (math.isfinite(v) and MIN_ABSTAIN_BELOW <= v < 1):
        raise argparse.ArgumentTypeError(
            f"must be at least {MIN_ABSTAIN_BELOW} and below 1 (below {MIN_ABSTAIN_BELOW} a lesion more likely "
            "neoplastic than not could be called benign)")
    return v


def _crop_margin(value: str) -> float:
    v = float(value)
    if not (math.isfinite(v) and 0 <= v <= MAX_CROP_MARGIN):
        raise argparse.ArgumentTypeError(f"must be between 0 and {MAX_CROP_MARGIN:g} (the classifier refuses other values)")
    return v


def _input_size(value: str) -> int:
    v = int(value)
    if v < MIN_INPUT_SIZE:
        raise argparse.ArgumentTypeError(f"must be at least {MIN_INPUT_SIZE} pixels")
    return v


def _fraction(value: str) -> float:
    v = float(value)
    if not (math.isfinite(v) and 0 <= v <= 1):
        raise argparse.ArgumentTypeError("must be between 0 and 1")
    return v


def main(argv: Optional[list[str]] = None) -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data", type=Path, action="append", required=True,
                    help="dataset directory containing labels.csv (repeat to combine data sets)")
    ap.add_argument("--out", type=Path, required=True, help="output directory")
    ap.add_argument("--evaluate-only", type=Path, metavar="MODEL.onnx",
                    help="evaluate an exported model (with its .json sidecar) on all rows of --data")
    ap.add_argument("--backbone", choices=BACKBONES, default="resnet18")
    ap.add_argument("--modality", choices=[m.value for m in Modality], default=Modality.COLONOSCOPY.value,
                    help="recordings the model is for; sets the working size images are shrunk to (as in "
                         "analysis) and is checked at analysis time")
    ap.add_argument("--pretrained", action="store_true", help="start from torchvision ImageNet weights (download)")
    ap.add_argument("--weights", type=Path, help="local backbone state_dict, e.g. torchvision ImageNet .pth")
    ap.add_argument("--init", type=Path, help="full best.pt checkpoint to fine-tune (e.g. synthetic pre-training)")
    ap.add_argument("--size", type=_input_size, default=224, help=f"crop size in pixels (at least {MIN_INPUT_SIZE})")
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--weight-decay", type=float, default=0.05)
    ap.add_argument("--patience", type=int, default=8, help="early stopping patience in epochs")
    ap.add_argument("--crop-margin", type=_crop_margin, default=DEFAULT_MARGIN,
                    help=f"context around the lesion box, as a fraction of its larger side (0 to {MAX_CROP_MARGIN:g})")
    ap.add_argument("--abstain-below", type=_abstain_threshold, default=0.6,
                    help=f"abstain when the top calibrated probability is below this, {MIN_ABSTAIN_BELOW} to <1 "
                         "(written to the sidecar; --evaluate-only uses the sidecar's value)")
    ap.add_argument("--min-frame-agreement", type=_fraction, default=0.5,
                    help="written to the sidecar: abstain when fewer frames agree with the aggregate")
    ap.add_argument("--val-frac", type=float, default=0.15)
    ap.add_argument("--test-frac", type=float, default=0.15)
    ap.add_argument("--bootstrap", type=int, default=1000, help="bootstrap resamples for 95%% CIs (0 = off)")
    ap.add_argument("--name", help="model name for the sidecar (default lesion-cadx-BACKBONE)")
    ap.add_argument("--version", help="model version for the sidecar (default: the UTC creation time)")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--workers", type=int, default=2, help="data loader worker processes")
    args = ap.parse_args(argv)
    if args.evaluate_only:
        evaluate_only(args)
    else:
        train(args)


if __name__ == "__main__":
    sys.exit(main())
