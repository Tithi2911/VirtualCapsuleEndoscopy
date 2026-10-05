"""Second-look comparison against the original procedure report.

The endoscopist's report lists the lesions they found (with the time in the
recording, or a frame number). Any AI finding that does not line up with a
reported lesion is marked POTENTIALLY_MISSED for a second reader to review.

Report file format (JSON):

    {
      "procedure_id": "optional",
      "findings": [
        {"id": "R1", "time_s": 312.4, "location": "sigmoid", "note": "6 mm sessile polyp, removed"},
        {"id": "R2", "frame_index": 10234}
      ]
    }
"""

from __future__ import annotations

import json
from pathlib import Path

from .config import AnalysisConfig
from .models import Finding, FindingStatus, ReportedFinding


def load_report(path: Path | str) -> list[ReportedFinding]:
    data = json.loads(Path(path).read_text())
    items = data["findings"] if isinstance(data, dict) else data
    out = []
    for i, item in enumerate(items):
        if item.get("time_s") is None and item.get("frame_index") is None:
            raise ValueError(f"Reported finding #{i + 1} needs 'time_s' or 'frame_index'")
        out.append(
            ReportedFinding(
                report_id=str(item.get("id", f"R{i + 1}")),
                time_s=item.get("time_s"),
                frame_index=item.get("frame_index"),
                location=item.get("location"),
                note=item.get("note"),
            )
        )
    return out


def _distance(finding: Finding, rep: ReportedFinding, cfg: AnalysisConfig) -> float | None:
    """How far the reported lesion is from the finding's visible interval, or None if out of tolerance."""
    if rep.frame_index is not None:
        lo, hi = finding.first.frame_index, finding.last.frame_index
        # Convert the frame tolerance using the finding's own frame/time ratio when possible.
        span_t = max(finding.duration_s, 1e-6)
        frames_per_s = (hi - lo) / span_t if hi > lo else 1.0
        tol = cfg.report_match_tolerance_s * frames_per_s
        x = rep.frame_index
    else:
        lo, hi = finding.first.timestamp_s, finding.last.timestamp_s
        tol = cfg.report_match_tolerance_s
        x = rep.time_s
    d = 0.0 if lo <= x <= hi else min(abs(x - lo), abs(x - hi))
    return d if d <= tol else None


def compare(findings: list[Finding], reported: list[ReportedFinding] | None, cfg: AnalysisConfig) -> list[ReportedFinding]:
    """Set each finding's status. Returns reported lesions the AI did not find (useful for audit)."""
    if reported is None:
        for f in findings:
            f.status = FindingStatus.UNREVIEWED
        return []

    # Greedy one-to-one matching, closest pairs first.
    pairs = []
    for fi, f in enumerate(findings):
        for ri, r in enumerate(reported):
            d = _distance(f, r, cfg)
            if d is not None:
                pairs.append((d, -f.peak_score, fi, ri))
    pairs.sort()
    used_f, used_r = set(), set()
    for _, _, fi, ri in pairs:
        if fi in used_f or ri in used_r:
            continue
        used_f.add(fi)
        used_r.add(ri)
        findings[fi].status = FindingStatus.REPORTED
        findings[fi].matched_report_id = reported[ri].report_id

    for fi, f in enumerate(findings):
        if fi not in used_f:
            f.status = FindingStatus.POTENTIALLY_MISSED
    return [r for ri, r in enumerate(reported) if ri not in used_r]
