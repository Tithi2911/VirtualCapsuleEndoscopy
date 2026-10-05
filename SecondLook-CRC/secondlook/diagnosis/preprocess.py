"""Lesion crop used by BOTH classifier training and inference.

Keeping a single implementation prevents train/serve skew, where the model is
trained on differently framed images from the ones it sees in use.
"""

from __future__ import annotations

import cv2
import numpy as np

DEFAULT_MARGIN = 0.25  # context around the lesion, as a fraction of its larger side


def square_crop_box(
    bbox: tuple[int, int, int, int], image_shape: tuple[int, ...], margin: float = DEFAULT_MARGIN
) -> tuple[int, int, int, int]:
    """Square box (x0, y0, x1, y1) around `bbox` (x, y, w, h), expanded by `margin` and clipped to the image."""
    h, w = image_shape[:2]
    x, y, bw, bh = bbox
    side = max(bw, bh) * (1 + 2 * margin)
    side = max(8.0, min(side, float(max(h, w))))
    cx, cy = x + bw / 2, y + bh / 2
    x0 = int(round(cx - side / 2))
    y0 = int(round(cy - side / 2))
    x1 = int(round(cx + side / 2))
    y1 = int(round(cy + side / 2))
    return max(0, x0), max(0, y0), min(w, x1), min(h, y1)


def crop_lesion(
    image: np.ndarray,
    bbox: tuple[int, int, int, int] | None,
    size: int,
    margin: float = DEFAULT_MARGIN,
) -> np.ndarray:
    """Return a `size` x `size` BGR crop around the lesion (whole image if bbox is None).

    Non-square crops at the image border are padded by edge replication rather
    than stretched, so lesion shape is preserved.
    """
    if bbox is None:
        crop = image
    else:
        x0, y0, x1, y1 = square_crop_box(bbox, image.shape, margin)
        crop = image[y0:y1, x0:x1]
    ch, cw = crop.shape[:2]
    side = max(ch, cw)
    if ch != cw:
        top = (side - ch) // 2
        left = (side - cw) // 2
        crop = cv2.copyMakeBorder(crop, top, side - ch - top, left, side - cw - left, cv2.BORDER_REPLICATE)
    interp = cv2.INTER_AREA if side > size else cv2.INTER_LINEAR
    return cv2.resize(crop, (size, size), interpolation=interp)


def bbox_from_mask(mask: np.ndarray) -> tuple[int, int, int, int] | None:
    """Bounding box (x, y, w, h) of all non-zero pixels, or None for an empty mask."""
    ys, xs = np.nonzero(mask)
    if xs.size == 0:
        return None
    return int(xs.min()), int(ys.min()), int(xs.max() - xs.min() + 1), int(ys.max() - ys.min() + 1)


def to_model_input(crops_bgr: list[np.ndarray], mean, std) -> np.ndarray:
    """Stack BGR uint8 crops into a normalised NCHW float32 RGB batch."""
    rgb = np.stack([cv2.cvtColor(c, cv2.COLOR_BGR2RGB) for c in crops_bgr]).astype(np.float32) / 255.0
    rgb = (rgb - np.asarray(mean, np.float32)) / np.asarray(std, np.float32)
    return np.ascontiguousarray(rgb.transpose(0, 3, 1, 2))
