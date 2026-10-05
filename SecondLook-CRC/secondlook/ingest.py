"""Load recordings from any common source: a video file, a folder of frames, or single images.

Vendor-neutral by design ("universal compatible"): anything OpenCV can decode is
accepted. Capsule vendors usually export frame sequences or AVI/MP4 from their
reading software; colonoscopy stacks export MP4/AVI or DICOM video. DICOM support
is a roadmap item (see docs/DESIGN.md).
"""

from __future__ import annotations

import hashlib
import math
import re
from pathlib import Path
from typing import Iterator

import cv2
import numpy as np

from .models import Frame

VIDEO_EXTS = {".mp4", ".avi", ".mov", ".mkv", ".mpg", ".mpeg", ".wmv", ".m4v"}
IMAGE_EXTS = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}


def _natural_key(path: Path):
    return [int(t) if t.isdigit() else t.lower() for t in re.split(r"(\d+)", path.name)]


def file_sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def input_fingerprint(path: Path) -> str:
    """SHA-256 of the input (or of all images in a folder), recorded in the audit log."""
    path = Path(path)
    if path.is_file():
        return file_sha256(path)
    h = hashlib.sha256()
    for p in sorted(_image_files(path), key=_natural_key):
        h.update(p.name.encode())
        h.update(file_sha256(p).encode())
    return h.hexdigest()


def _image_files(folder: Path) -> list[Path]:
    return [p for p in folder.iterdir() if p.suffix.lower() in IMAGE_EXTS]


def resize_to_working_size(image: np.ndarray, working_size: int) -> np.ndarray:
    """Shrink `image` so its longest side is `working_size` (never enlarged). Every analysed
    frame goes through this, and the lesion classifier's training images do too."""
    h, w = image.shape[:2]
    scale = working_size / max(h, w)
    if scale >= 1.0:
        return image
    return cv2.resize(image, (round(w * scale), round(h * scale)), interpolation=cv2.INTER_AREA)



def load_frames(
    path: Path | str,
    analysis_fps: float,
    working_size: int,
    image_sequence_fps: float = 1.0,
) -> Iterator[Frame]:
    """Yield frames to analyse, subsampled to roughly `analysis_fps`.

    For image folders there is no embedded timing, so frames are assumed to be
    `image_sequence_fps` apart (set it to the capsule's capture rate).
    """
    path = Path(path)
    if path.is_dir():
        files = sorted(_image_files(path), key=_natural_key)
        if not files:
            raise ValueError(f"No images found in {path}")
        yield from _from_images(files, analysis_fps, working_size, image_sequence_fps)
    elif path.suffix.lower() in IMAGE_EXTS:
        yield from _from_images([path], analysis_fps, working_size, image_sequence_fps)
    elif path.suffix.lower() in VIDEO_EXTS:
        yield from _from_video(path, analysis_fps, working_size)
    else:
        raise ValueError(f"Unsupported input: {path}")


def _step(source_fps: float, analysis_fps: float) -> int:
    if not math.isfinite(analysis_fps) or analysis_fps >= source_fps:
        return 1
    return max(1, round(source_fps / analysis_fps))


def _from_images(files, analysis_fps, working_size, sequence_fps) -> Iterator[Frame]:
    step = _step(sequence_fps, analysis_fps)
    for i in range(0, len(files), step):
        image = cv2.imread(str(files[i]), cv2.IMREAD_COLOR)
        if image is None:
            continue
        yield Frame(i, i / sequence_fps, resize_to_working_size(image, working_size), files[i].name)


def _from_video(path: Path, analysis_fps: float, working_size: int) -> Iterator[Frame]:
    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        raise ValueError(f"Could not open video {path}")
    try:
        fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
        step = _step(fps, analysis_fps)
        index = 0
        while True:
            ok = cap.grab()
            if not ok:
                break
            if index % step == 0:
                ok, image = cap.retrieve()
                if ok:
                    yield Frame(index, index / fps, resize_to_working_size(image, working_size), path.name)
            index += 1
    finally:
        cap.release()
