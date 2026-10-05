"""Link per-frame detections into findings.

A lesion is usually visible over many consecutive frames. Grouping detections
means the reviewer sees one finding per lesion (with how long it was in view)
instead of hundreds of boxes, and single-frame noise can be suppressed.
"""

from __future__ import annotations

import math

from .config import AnalysisConfig
from .models import Detection, Finding


def iou(a: tuple[int, int, int, int], b: tuple[int, int, int, int]) -> float:
    ax, ay, aw, ah = a
    bx, by, bw, bh = b
    ix = max(0, min(ax + aw, bx + bw) - max(ax, bx))
    iy = max(0, min(ay + ah, by + bh) - max(ay, by))
    inter = ix * iy
    union = aw * ah + bw * bh - inter
    return inter / union if union else 0.0


def _centre_distance(a, b, diag: float) -> float:
    ax, ay = a[0] + a[2] / 2, a[1] + a[3] / 2
    bx, by = b[0] + b[2] / 2, b[1] + b[3] / 2
    return math.hypot(ax - bx, ay - by) / diag


class Tracker:
    def __init__(self, cfg: AnalysisConfig):
        self.cfg = cfg
        self.active: list[list[Detection]] = []
        self.closed: list[list[Detection]] = []

    def update(self, detections: list[Detection], timestamp_s: float, frame_shape) -> None:
        diag = math.hypot(frame_shape[0], frame_shape[1])
        # Close tracks that have not been seen recently.
        still_active = []
        for track in self.active:
            if timestamp_s - track[-1].timestamp_s > self.cfg.track_max_gap_s:
                # Too-short tracks are dropped now so their pixel data is freed early.
                if len(track) >= self.cfg.min_track_frames:
                    self.closed.append(track)
            else:
                still_active.append(track)
        self.active = still_active

        claimed: set[int] = set()
        for det in sorted(detections, key=lambda d: -d.score):
            best, best_score = None, 0.0
            for ti, track in enumerate(self.active):
                if ti in claimed:
                    continue
                last = track[-1].bbox
                overlap = iou(last, det.bbox)
                near = _centre_distance(last, det.bbox, diag) < 0.12
                if overlap >= self.cfg.track_iou or near:
                    s = overlap + (0.01 if near else 0.0)
                    if best is None or s > best_score:
                        best, best_score = ti, s
            if best is None:
                self.active.append([det])
                claimed.add(len(self.active) - 1)
            else:
                self._append(self.active[best], det)
                claimed.add(best)

    @staticmethod
    def _append(track: list[Detection], det: Detection) -> None:
        # Keep pixel data only for the highest-scoring detection in each track.
        current_best = max(track, key=lambda d: d.score)
        if det.score > current_best.score:
            current_best.mask = current_best.heatmap = None
        else:
            det.mask = det.heatmap = None
        track.append(det)

    def findings(self) -> list[Finding]:
        tracks = self.closed + self.active
        kept = [t for t in tracks if len(t) >= self.cfg.min_track_frames]
        kept.sort(key=lambda t: t[0].timestamp_s)
        return [Finding(finding_id=f"F{i + 1:03d}", detections=t) for i, t in enumerate(kept)]
