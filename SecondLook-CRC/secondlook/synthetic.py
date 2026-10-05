"""Procedurally generated demo recordings and lesion images with known ground truth.

These are simple 2D renderings for exercising and testing the software end to
end, not training data. Realistic synthetic training data (3D colon anatomy,
capsule optics, lighting, polyps with labels) comes from the VR-Caps Unity
simulator in this repository - see docs/DATASETS.md.

Lesions can also be drawn as one of the three diagnostic categories
(secondlook.diagnosis.taxonomy), loosely after their white-light appearance:
small, pale, smooth hyperplastic-like polyps (benign); redder, lobulated
adenoma-like polyps with a tubular surface pattern (precancerous); and large,
ragged, dark red masses with a central fibrin-covered ulcer (cancerous). They
let the optical-diagnosis path - dataset loading, patient-grouped splits,
training, export, inference and reporting - be tested without patient data. A
model trained on them has learnt these drawings, not pathology.
"""

from __future__ import annotations

import csv
import json
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Sequence

import cv2
import numpy as np

from .diagnosis.taxonomy import CATEGORIES, DiagnosticCategory, category_for

H, W = 360, 480
MUCOSA_BGR = np.array([115, 135, 205], np.float32)

# Nominal lesion radius in pixels at 480 x 360, per category. Outlines vary around it.
LESION_RADIUS_PX: dict[str, tuple[float, float]] = {
    DiagnosticCategory.BENIGN.value: (10.0, 18.0),
    DiagnosticCategory.PRECANCEROUS.value: (16.0, 30.0),
    DiagnosticCategory.CANCEROUS.value: (28.0, 45.0),
}
# Colour (BGR) of each kind at full contrast, before shading; the lesion's contrast blends it with the mucosa.
LESION_TONE_BGR: dict[str, tuple[float, float, float]] = {
    DiagnosticCategory.BENIGN.value: (136, 130, 224),  # paler, whitish pink
    DiagnosticCategory.PRECANCEROUS.value: (78, 86, 204),  # redder and browner
    DiagnosticCategory.CANCEROUS.value: (50, 54, 160),  # dark red
}
# Atypical lesions are drawn part-way towards this category, as real lesions overlap in
# appearance (diminutive adenoma vs hyperplastic polyp, early cancer within an adenoma).
NEIGHBOUR_KIND = {
    DiagnosticCategory.BENIGN.value: DiagnosticCategory.PRECANCEROUS.value,
    DiagnosticCategory.PRECANCEROUS.value: DiagnosticCategory.BENIGN.value,
    DiagnosticCategory.CANCEROUS.value: DiagnosticCategory.PRECANCEROUS.value,
}
# Rows used in a recording, above and below the central lumen.
LESION_ROWS = (95, 265)


@dataclass
class SyntheticPolyp:
    canvas_x: int
    y: int
    radius: int
    category: Optional[str] = None


def _mucosa(width: int, rng: np.random.Generator, height: int = H, low_res: int = 1) -> np.ndarray:
    """Pink mucosa with smooth colour variation, fine texture and thin vessels.

    low_res > 1 computes the very smooth colour field at reduced resolution and draws
    float32 noise, which is much faster for many images; the default keeps the demo
    recordings identical from one release to the next.
    """
    fast = low_res > 1
    base = np.empty((height, width, 3), np.float32)
    base[:] = MUCOSA_BGR  # BGR pink mucosa
    if fast:
        small = rng.standard_normal((max(1, height // low_res), max(1, width // low_res)), dtype=np.float32)
        low = cv2.resize(cv2.GaussianBlur(small, (0, 0), 25 / low_res), (width, height), interpolation=cv2.INTER_CUBIC)
    else:
        low = cv2.GaussianBlur(rng.normal(0, 1, (height, width)).astype(np.float32), (0, 0), 25)
    low /= np.abs(low).max() + 1e-6
    base += low[..., None] * np.array([10, 14, 18], np.float32)
    if fast:
        noise = rng.standard_normal((height, width), dtype=np.float32) * 6
    else:
        noise = rng.normal(0, 6, (height, width)).astype(np.float32)
    fine = cv2.GaussianBlur(noise, (0, 0), 1.2)
    base += fine[..., None]
    # Thin, elongated vessels: red but not compact, so a good detector should ignore them.
    for _ in range(width * height // (40 * H)):
        pts = np.cumsum(rng.normal(0, 8, (12, 2)), axis=0) + [rng.uniform(0, width), rng.uniform(0, height)]
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


# ----------------------------------------------------------------------------- lesions by category


def _smooth_noise(shape: tuple[int, int], sigma: float, rng: np.random.Generator) -> np.ndarray:
    """Zero-mean noise blurred to `sigma` pixels, scaled to a peak of 1."""
    n = cv2.GaussianBlur(rng.normal(0, 1, shape).astype(np.float32), (0, 0), max(0.5, sigma))
    return n / (np.abs(n).max() + 1e-6)


def _outline(phi: np.ndarray, r: float, harmonics: dict[int, float], rng: np.random.Generator,
             floor: float = 0.55) -> np.ndarray:
    """Border radius per angle: a circle perturbed by harmonics. A few small ones give a round
    border, several mid-order ones give lobules, many give a ragged, irregular border."""
    rb = np.ones_like(phi)
    for k, a in harmonics.items():
        rb += a * np.cos(k * phi + rng.uniform(0, 2 * np.pi))
    return r * np.maximum(rb, floor)


def _points_inside(d: np.ndarray, limit: float, count: int, rng: np.random.Generator) -> np.ndarray:
    """`count` random (x, y) points, in patch pixels, where the normalised radius d < limit."""
    idx = np.flatnonzero(d.ravel() < limit)
    if idx.size == 0:
        return np.zeros((0, 2), np.float32)
    ys, xs = np.unravel_index(rng.choice(idx, count), d.shape)
    return np.stack([xs, ys], axis=1) + rng.uniform(-0.5, 0.5, (count, 2))


def _strokes(shape: tuple[int, int], paths: list[np.ndarray], thickness: int) -> np.ndarray:
    """Anti-aliased polylines as a 0..1 map."""
    img = np.zeros(shape, np.uint8)
    if paths:
        cv2.polylines(img, [np.round(p * 4).astype(np.int32) for p in paths], False, 255, thickness, cv2.LINE_AA, shift=2)
    return img.astype(np.float32) / 255.0


# Surface painters. `tone` is the lesion's base colour, `w` its typicality (1 = typical;
# lower values weaken the features that set the category apart).


def _benign_surface(r, d, s, R, tone, contrast, w, rng):
    """Hyperplastic-like (NICE type 1-like): close to mucosa colour but paler, smooth, low and
    regular, with faint dots of uniform size and at most an isolated lacy vessel."""
    colour = np.empty((*d.shape, 3), np.float32)
    colour[:] = tone + rng.normal(0, 3, 3)
    spacing = rng.uniform(3.5, 5.0)
    n = int(np.ceil(2 * R / spacing)) + 2
    j, i = np.mgrid[-n:n + 1, -n:n + 1].astype(np.float32)
    gx, gy = (i + 0.5 * (j % 2)) * spacing, j * spacing * np.sqrt(3) / 2
    a = rng.uniform(0, np.pi)
    px = gx * np.cos(a) - gy * np.sin(a) + R + rng.normal(0, 0.25, gx.shape)
    py = gx * np.sin(a) + gy * np.cos(a) + R + rng.normal(0, 0.25, gx.shape)
    dots = np.zeros(d.shape, np.uint8)
    for x, y in zip(px.ravel(), py.ravel()):
        if 0 <= x <= 2 * R and 0 <= y <= 2 * R:
            cv2.circle(dots, (int(round(x * 4)), int(round(y * 4))), 4, 255, -1, cv2.LINE_AA, shift=2)
    dots = cv2.GaussianBlur(dots.astype(np.float32) / 255.0, (0, 0), 0.5)
    spot = np.array([16, 16, 10], np.float32) if rng.random() < 0.65 else np.array([-14, -14, -9], np.float32)
    colour += dots[..., None] * spot * rng.uniform(0.5, 0.8) * contrast * (0.4 + 0.6 * w)
    if rng.random() < 0.5:
        start = _points_inside(d, 0.6, 1, rng)
        if len(start):
            path = start[0] + np.cumsum(rng.normal(0, 2.5, (6, 2)), axis=0)
            colour -= _strokes(d.shape, [path], 1)[..., None] * np.array([18, 30, 6], np.float32) * contrast
    z = (0.16 + 0.3 * (1 - w)) * r * s
    return colour, z


def _adenoma_surface(r, d, s, R, tone, contrast, w, rng):
    """Adenoma-like (NICE type 2-like): redder and browner than mucosa, lobulated, with short
    tubular or branched whitish structures outlined by brown vessels."""
    colour = np.empty((*d.shape, 3), np.float32)
    colour[:] = tone + rng.normal(0, 4, 3)
    colour += _smooth_noise(d.shape, r / 4, rng)[..., None] * np.array([5, 6, 10], np.float32)

    count = max(12, int(np.pi * r * r / 15))
    starts = _points_inside(d, 0.93, count, rng)
    radial = np.arctan2(starts[:, 1] - R, starts[:, 0] - R)
    angles = radial + rng.normal(0, 0.7, count)  # pits tend to run outwards from the centre
    lengths = rng.uniform(3, 7, count)
    vessels, pits = [], []
    for p0, ang, length in zip(starts, angles, lengths):
        u = np.array([np.cos(ang), np.sin(ang)])
        nrm = np.array([-u[1], u[0]])
        p2 = p0 + length * u
        p1 = (p0 + p2) / 2 + nrm * rng.normal(0, 0.8)
        vessels.append(np.stack([p0, p1, p2]))
        pits.append(np.stack([p0, p1, p2]) + nrm * 1.6)
        if rng.random() < 0.35:
            b = ang + rng.choice([-1.0, 1.0]) * rng.uniform(0.5, 1.0)
            vessels.append(np.stack([p1, p1 + 0.6 * length * np.array([np.cos(b), np.sin(b)])]))
    white = cv2.GaussianBlur(_strokes(d.shape, pits, 2), (0, 0), 0.6)
    brown = _strokes(d.shape, vessels, 1)
    visible = contrast * (0.3 + 0.7 * w)
    colour += white[..., None] * np.array([20, 20, 12], np.float32) * rng.uniform(0.6, 1.0) * visible
    k = rng.uniform(0.55, 0.8) * visible * brown
    colour = colour * (1 - k[..., None]) + np.array([42, 60, 122], np.float32) * k[..., None]
    z = (0.25 + 0.3 * w) * r * s**0.6 + 0.1 * w * r * _smooth_noise(d.shape, r / 5, rng) * s
    return colour, z


def _cancer_surface(r, d, s, R, tone, contrast, w, rng):
    """Cancer-like (NICE type 3-like): dark red and heterogeneous, a heaped-up edge, a central
    depressed ulcer covered by whitish-yellow fibrin, and a disrupted or absent surface pattern.
    Atypical ones (early cancer within an adenoma) have a smaller ulcer and less heterogeneity."""
    colour = np.empty((*d.shape, 3), np.float32)
    colour[:] = tone + rng.normal(0, 5, 3)
    hetero = 0.5 + 0.5 * w
    colour += _smooth_noise(d.shape, r / 5, rng)[..., None] * np.array([12, 10, 34], np.float32) * hetero
    colour += _smooth_noise(d.shape, r / 9, rng)[..., None] * np.array([10, -4, 4], np.float32) * hetero

    # Disrupted pattern: sparse, irregular dark fragments and amorphous blotches.
    count = max(6, int(np.pi * r * r / 110))
    paths = [p + np.cumsum(rng.normal(0, 2.5, (int(rng.integers(3, 6)), 2)), axis=0)
             for p in _points_inside(d, 0.9, count, rng)]
    frag = np.maximum(_strokes(d.shape, paths[: count // 2], 1), _strokes(d.shape, paths[count // 2:], 2))
    blot = cv2.GaussianBlur(np.clip((_smooth_noise(d.shape, r / 8, rng) - 0.4) * 3, 0, 1), (0, 0), 1.0)
    dark = np.clip(0.7 * frag + 0.5 * blot, 0, 0.8)[..., None] * w
    colour = colour * (1 - dark) + np.array([34, 36, 92], np.float32) * dark

    # Central ulcer with a haemorrhagic edge.
    yy, xx = np.mgrid[-R:R + 1, -R:R + 1].astype(np.float32)
    ox, oy = rng.normal(0, 0.12 * r, 2)
    rho_u = np.hypot(xx - ox, yy - oy)
    ub = _outline(np.arctan2(yy - oy, xx - ox), r * rng.uniform(0.28, 0.5) * w,
                  {k: rng.uniform(0.04, 0.12) for k in range(2, 9)}, rng, floor=0.6)
    ulcer = np.clip((ub - rho_u) / 2.0, 0, 1)
    rim = np.clip(1 - np.abs(rho_u - ub) / (0.25 * ub), 0, 1) * 0.5
    colour = colour * (1 - rim[..., None]) + np.array([44, 40, 150], np.float32) * rim[..., None]
    fibrin = np.empty_like(colour)
    fibrin[:] = np.array([150, 196, 222], np.float32) + rng.normal(0, 6, 3)
    fibrin += _smooth_noise(d.shape, 1.0, rng)[..., None] * 14
    fibrin += _smooth_noise(d.shape, r / 6, rng)[..., None] * np.array([-14, -8, 4], np.float32)
    for x, y in _points_inside(rho_u / ub, 0.8, int(rng.integers(0, 4)), rng):
        cv2.circle(fibrin, (int(x), int(y)), int(rng.integers(1, 3)), (40, 35, 140), -1, cv2.LINE_AA)
    colour = colour * (1 - ulcer[..., None]) + fibrin * ulcer[..., None]

    rim_height = np.exp(-(((d - 0.75) / 0.18) ** 2))
    crater = cv2.GaussianBlur(ulcer, (0, 0), 2.0)
    z = 0.45 * r * s**0.5 + 0.16 * w * r * rim_height * (d < 1) + 0.07 * r * _smooth_noise(d.shape, r / 6, rng) * s
    z -= 0.32 * w * r * crater
    return colour, z


_SURFACES = {
    DiagnosticCategory.BENIGN.value: _benign_surface,
    DiagnosticCategory.PRECANCEROUS.value: _adenoma_surface,
    DiagnosticCategory.CANCEROUS.value: _cancer_surface,
}


def _shade(z: np.ndarray, rng: np.random.Generator) -> tuple[np.ndarray, np.ndarray]:
    """Diffuse shading (1.0 on flat mucosa) and a specular term for a light near the scope axis."""
    gy, gx = np.gradient(z)
    n = np.stack([-gx, -gy, np.ones_like(z)], axis=-1)
    n /= np.linalg.norm(n, axis=-1, keepdims=True)
    tilt, b = rng.uniform(0.25, 0.55), rng.uniform(0, 2 * np.pi)
    light = np.array([tilt * np.cos(b), tilt * np.sin(b), 1.0])
    light /= np.linalg.norm(light)
    shade = 0.4 + 0.6 * np.clip(n @ light, 0, None) / light[2]
    half = light + [0.0, 0.0, 1.0]
    half /= np.linalg.norm(half)
    spec = np.clip((n @ half - half[2]) / (1 - half[2]), 0, 1) ** 24
    return shade.astype(np.float32), spec.astype(np.float32)


def _border_style(kind: str, r: float, rng: np.random.Generator) -> tuple[dict[int, float], float]:
    """Outline harmonics and border softness (px) for a lesion of radius r."""
    if kind == DiagnosticCategory.BENIGN.value:
        # Round and regular, with a soft, low-contrast border.
        return {2: rng.uniform(0, 0.08), 3: rng.uniform(0, 0.03)}, 0.3 * r
    if kind == DiagnosticCategory.PRECANCEROUS.value:
        return {2: rng.uniform(0.02, 0.08), **{k: rng.uniform(0.03, 0.065) for k in range(3, 6)}}, 2.5
    if kind == DiagnosticCategory.CANCEROUS.value:
        harmonics = {k: rng.uniform(0.3, 1.0) * 0.13 / k**0.6 for k in range(2, 10)}
        harmonics.update({k: rng.uniform(0, 0.012) for k in range(10, 25)})
        return harmonics, 1.5
    raise ValueError(f"Unknown lesion kind {kind!r}; expected one of {CATEGORIES}")


def _lesion_radius(kind: str, typicality: float, rng: np.random.Generator) -> float:
    lo, hi = LESION_RADIUS_PX[kind]
    if typicality < 1:
        nlo, nhi = LESION_RADIUS_PX[NEIGHBOUR_KIND[kind]]
        lo, hi = typicality * lo + (1 - typicality) * nlo, typicality * hi + (1 - typicality) * nhi
    return float(rng.uniform(lo, hi))


def _render_lesion(kind: str, radius: float, rng: np.random.Generator,
                   typicality: float = 1.0) -> tuple[np.ndarray, np.ndarray]:
    """Shaded BGR colour (float32) and 0..1 coverage of one lesion on a square patch centred on it.

    typicality < 1 draws the lesion part-way towards NEIGHBOUR_KIND[kind] (outline, colour and
    surface features); it is still a lesion of `kind`.
    """
    r, w = float(radius), float(np.clip(typicality, 0.0, 1.0))
    harmonics, edge = _border_style(kind, r, rng)
    contrast = rng.uniform(0.7, 1.0)
    target = np.array(LESION_TONE_BGR[kind], np.float32)
    if w < 1:
        other = NEIGHBOUR_KIND[kind]
        other_harmonics, other_edge = _border_style(other, r, rng)
        harmonics = {k: w * harmonics.get(k, 0.0) + (1 - w) * other_harmonics.get(k, 0.0)
                     for k in sorted(harmonics.keys() | other_harmonics.keys())}
        edge = w * edge + (1 - w) * other_edge
        target = w * target + (1 - w) * np.array(LESION_TONE_BGR[other], np.float32)
    tone = MUCOSA_BGR + contrast * (target - MUCOSA_BGR)

    R = int(np.ceil(r * (1 + sum(harmonics.values())))) + 3
    yy, xx = np.mgrid[-R:R + 1, -R:R + 1].astype(np.float32)
    rho = np.hypot(xx, yy)
    rb = _outline(np.arctan2(yy, xx) - rng.uniform(0, 2 * np.pi), r, harmonics, rng)
    d = rho / rb
    s = np.clip(1 - d**2, 0, 1)
    alpha = np.clip((rb - rho) / edge, 0, 1).astype(np.float32)

    colour, z = _SURFACES[kind](r, d, s, R, tone, contrast, w, rng)
    shade, spec = _shade(z, rng)
    colour = colour * shade[..., None] + 120 * spec[..., None]
    colour += cv2.GaussianBlur(rng.normal(0, 5, colour.shape).astype(np.float32), (0, 0), 0.8)
    return colour.astype(np.float32), alpha


def _extent(alpha: np.ndarray) -> float:
    """Largest distance from the patch centre to a visibly covered pixel."""
    R = alpha.shape[0] // 2
    ys, xs = np.nonzero(alpha >= 0.5)
    return float(np.hypot(xs - R, ys - R).max()) if xs.size else 1.0


def _paste(canvas: np.ndarray, colour: np.ndarray, alpha: np.ndarray, cx: int, cy: int,
           mask: Optional[np.ndarray] = None) -> None:
    """Alpha-blend a lesion patch centred at (cx, cy), clipped to the canvas; mark it in `mask`."""
    R = colour.shape[0] // 2
    ch, cw = canvas.shape[:2]
    y0, x0 = cy - R, cx - R
    ys, xs, ye, xe = max(0, y0), max(0, x0), min(ch, y0 + 2 * R + 1), min(cw, x0 + 2 * R + 1)
    if ye <= ys or xe <= xs:
        return
    a = alpha[ys - y0:ye - y0, xs - x0:xe - x0]
    canvas[ys:ye, xs:xe] = canvas[ys:ye, xs:xe] * (1 - a[..., None]) + colour[ys - y0:ye - y0, xs - x0:xe - x0] * a[..., None]
    if mask is not None:
        mask[ys:ye, xs:xe] |= a >= 0.5


def _lesion_categories(kinds: Sequence[str]) -> list[str]:
    """Category values for histology or category labels (category_for raises KeyError if unknown)."""
    cats = [category_for(k).value for k in kinds]
    if not cats:
        raise ValueError("Give at least one lesion kind")
    return cats


def _spread_centres(k: int, duration: float, half_view: float, blur: Optional[tuple[float, float]]) -> list[float]:
    """Times (s) at which k lesions are centred in view, spread evenly over the stretches where
    each is fully in view and the recording is not blurred, so every lesion can be reviewed."""
    lo, hi = half_view + 0.5, duration - half_view - 0.5
    if hi <= lo:
        return [duration / 2] * k
    segments = [(lo, hi)]
    if blur:
        segments = [(lo, min(hi, blur[0] - half_view)), (max(lo, blur[1] + half_view), hi)]
        segments = [(a, b) for a, b in segments if b > a] or [(lo, hi)]
    total = sum(b - a for a, b in segments)
    marks = [total / 2] if k == 1 else list(np.linspace(0, total, k))
    centres = []
    for m in marks:
        for a, b in segments:
            if m <= b - a + 1e-9:
                centres.append(a + m)
                break
            m -= b - a
    return centres


def _place_lesions(canvas: np.ndarray, kinds: list[str], n_frames: int, speed: float, fps: float,
                   blur: Optional[tuple[float, float]], rng: np.random.Generator) -> list[SyntheticPolyp]:
    duration = n_frames / fps
    centres = _spread_centres(len(kinds), duration, W / 2 / (speed * fps), blur)
    polyps = []
    for i, (kind, t) in enumerate(zip(kinds, centres)):
        colour, alpha = _render_lesion(kind, _lesion_radius(kind, 1.0, rng), rng)
        ext = _extent(alpha)
        y = int(np.clip(LESION_ROWS[i % 2] + rng.integers(-10, 11), ext + 4, H - ext - 4))
        cx = int(round(W / 2 + t * fps * speed))
        _paste(canvas, colour, alpha, cx, y)
        polyps.append(SyntheticPolyp(cx, y, int(np.ceil(ext)), kind))
    return polyps


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
    lesion_kinds: Sequence[str] | None = None,
) -> dict:
    """Render a recording plus ground truth. Returns the ground-truth dict (also written as JSON).

    as_frames=True writes a folder of PNGs (capsule-style export) instead of a video.

    By default two generic polyps are drawn. lesion_kinds, e.g. ["benign", "precancerous",
    "cancerous"], draws one lesion of each listed category instead, spread evenly over the
    unblurred part of the recording and alternating above and below the lumen; each truth
    entry then has a "category". At the default 40 s, three lesions fit only partly between
    the blurred stretch and the end; use duration_s=60 or blur_interval_s=None for more room.
    The mock procedure report mentions only the first lesion either way.
    """
    rng = np.random.default_rng(seed)
    n = int(duration_s * fps)
    speed = 6.0 if not as_frames else 20.0  # px/frame camera pan; capsules jump more between frames
    canvas_w = int(W + speed * n) + 10
    canvas = _mucosa(canvas_w, rng)

    if lesion_kinds is None:
        # Two polyps; the procedure report will only mention the first.
        polyps = [
            SyntheticPolyp(int(W + speed * n * 0.15), 95, 30),
            SyntheticPolyp(int(W + speed * n * 0.75), 265, 26),
        ]
        for p in polyps:
            _draw_polyp(canvas, p, rng)
    else:
        polyps = _place_lesions(canvas, _lesion_categories(lesion_kinds), n, speed, fps, blur_interval_s, rng)

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
            {"id": f"P{i + 1}", "visible_from_s": visible(p)[0], "visible_to_s": visible(p)[1], "radius_px": p.radius,
             **({"category": p.category} if p.category else {})}
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


# ----------------------------------------------------------------------------- labelled lesion images


def _lesion_view(scene: np.ndarray, mask: np.ndarray, ext: float, size: tuple[int, int], noise: np.ndarray,
                 rng: np.random.Generator) -> tuple[np.ndarray, np.ndarray]:
    """One photograph of the scene's lesion: new framing, distance, rotation, light, focus and noise.
    `noise` is a bank of unit Gaussian noise, at least `size` plus a margin, cropped at random."""
    w, h = size
    centre = (scene.shape[1] / 2, scene.shape[0] / 2)
    scale = min(rng.uniform(0.8, 1.25), (0.5 * min(w, h) - 6) / ext)
    tx = w / 2 + rng.uniform(-1, 1) * min(0.15 * w, max(0.0, w / 2 - ext * scale - 6))
    ty = h / 2 + rng.uniform(-1, 1) * min(0.15 * h, max(0.0, h / 2 - ext * scale - 6))
    M = cv2.getRotationMatrix2D(centre, rng.uniform(0, 360), scale)
    M[:, 2] += (tx - centre[0], ty - centre[1])
    img = cv2.warpAffine(scene, M, (w, h), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT_101)
    m = cv2.warpAffine(mask.astype(np.uint8) * 255, M, (w, h), flags=cv2.INTER_LINEAR) >= 128

    # Light falls off away from the scope axis, which is not always at the image centre.
    # It varies slowly, so it is computed on a coarse grid and scaled up.
    gh, gw = max(8, h // 8), max(8, w // 8)
    yy, xx = np.mgrid[0:gh, 0:gw].astype(np.float32)
    yy, xx = (yy + 0.5) * (h / gh), (xx + 0.5) * (w / gw)
    ax, ay = w * rng.uniform(0.35, 0.65), h * rng.uniform(0.35, 0.65)
    light = 1 - rng.uniform(0.25, 0.55) * (((xx - ax) / (w / 2)) ** 2 + ((yy - ay) / (h / 2)) ** 2) / 2
    if rng.random() < 0.4:  # dark lumen towards the side away from the lesion
        a = np.arctan2(h / 2 - ty, w / 2 - tx) if (tx, ty) != (w / 2, h / 2) else rng.uniform(0, 2 * np.pi)
        a += rng.normal(0, 0.3)
        lx, ly = w / 2 + 0.6 * w * np.cos(a), h / 2 + 0.6 * h * np.sin(a)
        light *= 1 - 0.85 * np.exp(-(((xx - lx) / (0.22 * w)) ** 2 + ((yy - ly) / (0.2 * h)) ** 2))
    light = cv2.resize(light * rng.uniform(0.8, 1.15), (w, h), interpolation=cv2.INTER_LINEAR)
    img = img * light[..., None] * (1 + rng.normal(0, 0.03, 3)).astype(np.float32)

    for _ in range(int(rng.integers(0, 4))):  # glints on wet mucosa
        cv2.circle(img, (int(rng.uniform(0, w)), int(rng.uniform(0, h))), int(rng.uniform(1, 4)), (255, 255, 255), -1, cv2.LINE_AA)
    ys, xs = np.nonzero(m)
    if xs.size and rng.random() < 0.6:  # and on the lesion surface
        i = int(rng.integers(xs.size))
        cv2.circle(img, (int(xs[i]), int(ys[i])), int(rng.uniform(1, 3)), (255, 255, 255), -1, cv2.LINE_AA)
    if rng.random() < 0.35:
        img = cv2.GaussianBlur(img, (0, 0), rng.uniform(0.6, 1.6))
    sigma = rng.uniform(1, 4)
    oy, ox = int(rng.integers(0, noise.shape[0] - h + 1)), int(rng.integers(0, noise.shape[1] - w + 1))
    img = np.clip(img + noise[oy:oy + h, ox:ox + w] * sigma, 0, 255).astype(np.uint8)
    if rng.random() < 0.3:  # video compression
        ok, buf = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, int(rng.integers(55, 91))])
        img = cv2.imdecode(buf, cv2.IMREAD_COLOR) if ok else img
    return img, m


def generate_lesion_dataset(
    out_dir: Path | str,
    n_per_class: int = 60,
    views_per_lesion: int = 3,
    size: tuple[int, int] = (256, 256),
    seed: int = 0,
    kinds: Sequence[str] | None = None,
    atypical_fraction: float = 0.15,
) -> dict:
    """Write labelled synthetic lesion images in the training/train_classifier.py format.

    Each synthetic patient has one lesion, photographed `views_per_lesion` times with a
    different framing, distance, rotation, lighting, focus and noise, as frames of one
    lesion in a real video are. Views of a lesion are near-duplicates, so the split must
    keep a patient's images together; the patient_id column makes that possible.

    A fraction of lesions (`atypical_fraction`) is drawn part-way towards the neighbouring
    category but keeps its true label. Without that overlap the classes are perfectly
    separable, so calibration, abstention and the error metrics could never be exercised.

    Writes out_dir/images/*.png, out_dir/masks/*.png (white = lesion), out_dir/labels.csv
    (image,label,patient_id,mask,appearance) and out_dir/dataset.json. `size` is (width,
    height) at the pixel scale of the 480 x 360 demo recordings. Returns the summary also
    written to dataset.json. For software testing and synthetic pre-training only.
    """
    if n_per_class < 1 or views_per_lesion < 1:
        raise ValueError("n_per_class and views_per_lesion must be at least 1")
    if not 0 <= atypical_fraction <= 1:
        raise ValueError("atypical_fraction must be between 0 and 1")
    w, h = int(size[0]), int(size[1])
    if min(w, h) < 64:
        raise ValueError("size must be at least 64 x 64 pixels")
    cats = _lesion_categories(kinds) if kinds is not None else list(CATEGORIES)
    cats = list(dict.fromkeys(cats))
    out_dir = Path(out_dir)
    (out_dir / "images").mkdir(parents=True, exist_ok=True)
    (out_dir / "masks").mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(seed)

    # Shuffled so patient numbers do not reveal the class.
    assignment = [c for c in cats for _ in range(n_per_class)]
    rng.shuffle(assignment)
    # Covers every framing of an unrotated view; the corners of rotated, zoomed-out views
    # are filled by reflection, which looks like more mucosa.
    half = int(np.ceil(0.65 * max(w, h) / 0.8)) + 4
    noise = rng.standard_normal((h + 64, w + 64, 3), dtype=np.float32)
    rows, writes = [], []
    atypical = {c: 0 for c in cats}
    # PNG encoding releases the GIL, so files are written while the next lesion is drawn.
    with ThreadPoolExecutor(max_workers=2) as pool:
        for i, kind in enumerate(assignment):
            pid = f"SYN{i + 1:05d}"
            scene = _mucosa(2 * half + 1, rng, height=2 * half + 1, low_res=4)
            typicality = rng.uniform(0.35, 0.65) if rng.random() < atypical_fraction else 1.0
            atypical[kind] += typicality < 1
            appearance = "typical" if typicality == 1 else "atypical"
            colour, alpha = _render_lesion(kind, _lesion_radius(kind, typicality, rng), rng, typicality)
            mask = np.zeros(scene.shape[:2], bool)
            _paste(scene, colour, alpha, half, half, mask)
            scene *= (1 + rng.normal(0, 0.04, 3)).astype(np.float32)  # each patient's mucosa looks a bit different
            ext = _extent(alpha)
            for v in range(views_per_lesion):
                img, m = _lesion_view(scene, mask, ext, (w, h), noise, rng)
                name = f"{pid}_v{v + 1}.png"
                for path, data in ((out_dir / "images" / name, img), (out_dir / "masks" / name, m.astype(np.uint8) * 255)):
                    writes.append((path, pool.submit(cv2.imwrite, str(path), data)))
                rows.append({"image": f"images/{name}", "label": kind, "patient_id": pid, "mask": f"masks/{name}",
                             "appearance": appearance})
    failed = [str(path) for path, done in writes if not done.result()]
    if failed:
        raise OSError(f"Could not write {len(failed)} image files, e.g. {failed[0]}")

    labels = out_dir / "labels.csv"
    with labels.open("w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=["image", "label", "patient_id", "mask", "appearance"])
        writer.writeheader()
        writer.writerows(rows)
    summary = {
        "description": "Procedurally generated lesion images for software testing and synthetic pre-training. "
                       "Not clinical data; labels are the category each image was drawn as, not histology.",
        # train_classifier.py reads this, so a model trained only on these drawings says so in its reports.
        "synthetic": True,
        "labels_csv": str(labels),
        "patients": len(assignment),
        "images": len(rows),
        "views_per_lesion": views_per_lesion,
        "size": [w, h],
        "seed": seed,
        "atypical_fraction": atypical_fraction,
        "per_class": {c: {"patients": n_per_class, "images": n_per_class * views_per_lesion,
                          "atypical_patients": atypical[c]} for c in cats},
    }
    (out_dir / "dataset.json").write_text(json.dumps({**summary, "labels_csv": labels.name}, indent=2))
    return summary
