"""Procedurally generated demo recordings with known ground truth.

These are simple 2D renderings for exercising and testing the software end to
end, not training data. Realistic synthetic training data (3D colon anatomy,
capsule optics, lighting, polyps with labels) comes from the VR-Caps Unity
simulator in this repository - see docs/DATASETS.md.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np

H, W = 360, 480


@dataclass
class SyntheticPolyp:
    canvas_x: int
    y: int
    radius: int


def _mucosa(width: int, rng: np.random.Generator) -> np.ndarray:
    base = np.empty((H, width, 3), np.float32)
    base[:] = (115, 135, 205)  # BGR pink mucosa
    low = cv2.GaussianBlur(rng.normal(0, 1, (H, width)).astype(np.float32), (0, 0), 25)
    low /= np.abs(low).max() + 1e-6
    base += low[..., None] * np.array([10, 14, 18], np.float32)
    fine = cv2.GaussianBlur(rng.normal(0, 6, (H, width)).astype(np.float32), (0, 0), 1.2)
    base += fine[..., None]
    # Thin, elongated vessels: red but not compact, so a good detector should ignore them.
    for _ in range(width // 40):
        pts = np.cumsum(rng.normal(0, 8, (12, 2)), axis=0) + [rng.uniform(0, width), rng.uniform(0, H)]
        cv2.polylines(base, [pts.astype(np.int32)], False, (70, 70, 185), 1, cv2.LINE_AA)
    return base


def _draw_polyp(canvas: np.ndarray, p: SyntheticPolyp, rng: np.random.Generator) -> None:
    r = p.radius
    yy, xx = np.mgrid[-r:r + 1, -r:r + 1].astype(np.float32)
    d = np.sqrt(xx**2 + yy**2) / r
    inside = d <= 1.0
    dome = np.sqrt(np.clip(1 - d**2, 0, 1))
    light = np.clip(0.55 + 0.45 * (-(xx + yy) / (r * 1.4)) * dome + 0.3 * dome, 0.3, 1.25)
    colour = np.array([85, 90, 215], np.float32)  # redder, more hyperaemic than mucosa
    patch = colour * light[..., None] + rng.normal(0, 9, (*d.shape, 3)).astype(np.float32)
    y0, x0 = p.y - r, p.canvas_x - r
    region = canvas[y0:y0 + 2 * r + 1, x0:x0 + 2 * r + 1]
    edge = np.clip((1.0 - d) * r / 2.0, 0, 1)[..., None]  # soft border
    region[:] = np.where(inside[..., None], region * (1 - edge) + patch * edge, region)


def _finish(view: np.ndarray, t: int, rng: np.random.Generator) -> np.ndarray:
    yy, xx = np.mgrid[0:H, 0:W].astype(np.float32)
    # Dark lumen near the centre, drifting slightly.
    cx, cy = W / 2 + 25 * np.sin(t / 30), H / 2 + 12 * np.cos(t / 40)
    lumen = np.exp(-(((xx - cx) / 55) ** 2 + ((yy - cy) / 45) ** 2))
    vignette = 1 - 0.55 * (((xx - W / 2) / (W / 2)) ** 2 + ((yy - H / 2) / (H / 2)) ** 2) / 2
    img = view * (vignette * (1 - 0.9 * lumen))[..., None]
    # A few specular highlights.
    for _ in range(3):
        cv2.circle(img, (int(rng.uniform(40, W - 40)), int(rng.uniform(40, H - 40))), int(rng.uniform(2, 5)), (255, 255, 255), -1)
    return np.clip(img, 0, 255).astype(np.uint8)


def generate(
    out_path: Path | str,
    duration_s: float = 40.0,
    fps: float = 10.0,
    blur_interval_s: tuple[float, float] | None = (20.0, 27.0),
    as_frames: bool = False,
    seed: int = 7,
) -> dict:
    """Render a recording plus ground truth. Returns the ground-truth dict (also written as JSON).

    as_frames=True writes a folder of PNGs (capsule-style export) instead of a video.
    """
    rng = np.random.default_rng(seed)
    n = int(duration_s * fps)
    speed = 6.0 if not as_frames else 20.0  # px/frame camera pan; capsules jump more between frames
    canvas_w = int(W + speed * n) + 10
    canvas = _mucosa(canvas_w, rng)

    # Two polyps; the procedure report will only mention the first.
    polyps = [
        SyntheticPolyp(int(W + speed * n * 0.15), 95, 30),
        SyntheticPolyp(int(W + speed * n * 0.75), 265, 26),
    ]
    for p in polyps:
        _draw_polyp(canvas, p, rng)

    out_path = Path(out_path)
    if as_frames:
        out_path.mkdir(parents=True, exist_ok=True)
        writer = None
    else:
        out_path.parent.mkdir(parents=True, exist_ok=True)
        writer = cv2.VideoWriter(str(out_path), cv2.VideoWriter_fourcc(*"MJPG"), fps, (W, H))
        if not writer.isOpened():
            raise RuntimeError("Could not open video writer")

    for t in range(n):
        x = int(t * speed)
        frame = _finish(canvas[:, x:x + W], t, rng)
        if blur_interval_s and blur_interval_s[0] <= t / fps < blur_interval_s[1]:
            frame = cv2.GaussianBlur(frame, (0, 0), 6)
        if writer:
            writer.write(frame)
        else:
            cv2.imwrite(str(out_path / f"frame_{t:05d}.png"), frame)
    if writer:
        writer.release()

    def visible(p: SyntheticPolyp):
        # Camera left edge x = t*speed; polyp fully in view while x <= cx-r and x+W >= cx+r.
        t0 = max(0.0, (p.canvas_x + p.radius - W) / speed)
        t1 = (p.canvas_x - p.radius) / speed
        return t0 / fps, t1 / fps

    truth = {
        "fps": fps,
        "polyps": [
            {"id": f"P{i + 1}", "visible_from_s": visible(p)[0], "visible_to_s": visible(p)[1], "radius_px": p.radius}
            for i, p in enumerate(polyps)
        ],
        "blur_interval_s": list(blur_interval_s) if blur_interval_s else None,
    }
    gt_path = (out_path if as_frames else out_path.with_suffix("")).with_name(out_path.stem + "_truth.json")
    gt_path.write_text(json.dumps(truth, indent=2))

    # Procedure report as an endoscopist would have written it: only the first polyp was noticed.
    first = truth["polyps"][0]
    report = {
        "procedure_id": "SYNTH-001",
        "findings": [
            {
                "id": "R1",
                "time_s": round((first["visible_from_s"] + first["visible_to_s"]) / 2, 1),
                "location": "sigmoid",
                "note": "sessile polyp, removed",
            }
        ],
    }
    gt_path.with_name(out_path.stem + "_report.json").write_text(json.dumps(report, indent=2))
    return truth
