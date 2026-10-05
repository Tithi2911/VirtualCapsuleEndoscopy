"""Write analysis results as machine-readable JSON plus a self-contained HTML report.

The HTML report is a single file (images embedded) so it can be attached to a
patient record, shared inside an MDT, or archived for audit without a server.
It includes reviewer controls (confirm / reject / unsure + comment per finding)
that export a decisions JSON: this is the second-reader sign-off and, with
consent and governance, the labelled feedback used to retrain the model.
"""

from __future__ import annotations

import base64
import html
import json
from datetime import datetime, timezone
from pathlib import Path

import cv2

from . import INTENDED_USE_NOTICE, __version__
from .models import FindingStatus, to_jsonable
from .pipeline import AnalysisResult

STATUS_LABEL = {
    FindingStatus.POTENTIALLY_MISSED: "Potentially missed",
    FindingStatus.REPORTED: "Matches report",
    FindingStatus.UNREVIEWED: "Not compared",
}


def _fmt_time(t: float) -> str:
    m, s = divmod(t, 60)
    return f"{int(m):02d}:{s:04.1f}"


def result_dict(result: AnalysisResult) -> dict:
    return {
        "software": {"name": "SecondLook-CRC", "version": __version__},
        "intended_use_notice": INTENDED_USE_NOTICE,
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "input": result.input_path,
        "modality": result.config.modality.value,
        "config": to_jsonable(result.config),
        "detector": {"name": result.detector, "version": result.detector_version},
        "characteriser": result.characteriser,
        "summary": {
            "duration_s": result.duration_s,
            "frames_analysed": result.frames_analysed,
            "frames_informative": result.frames_informative,
            "findings": len(result.findings),
            "potentially_missed": sum(f.status == FindingStatus.POTENTIALLY_MISSED for f in result.findings),
            "blind_segments": len(result.blind_segments),
            "blind_time_s": sum(b.duration_s for b in result.blind_segments),
            "report_supplied": result.report_supplied,
            "runtime_s": result.runtime_s,
        },
        "findings": [
            {
                "id": f.finding_id,
                "status": f.status.value,
                "matched_report_id": f.matched_report_id,
                "first_s": f.first.timestamp_s,
                "last_s": f.last.timestamp_s,
                "best_s": f.best.timestamp_s,
                "best_frame_index": f.best.frame_index,
                "peak_score": f.peak_score,
                "mean_score": f.mean_score,
                "frames": len(f.detections),
                "bbox_at_best": list(f.best.bbox),
                "evidence": f.best.evidence,
                "rationale": f.rationale,
                "characterisation": to_jsonable(f.characterisation),
            }
            for f in result.findings
        ],
        "blind_segments": to_jsonable(result.blind_segments),
        "reported_not_found_by_ai": to_jsonable(result.unmatched_reported),
    }


def _img_tag(image, alt: str, cls: str = "") -> str:
    ok, buf = cv2.imencode(".jpg", image, [cv2.IMWRITE_JPEG_QUALITY, 88])
    data = base64.b64encode(buf.tobytes()).decode() if ok else ""
    return f'<img class="{cls}" alt="{html.escape(alt)}" src="data:image/jpeg;base64,{data}">'


def _timeline_svg(result: AnalysisResult) -> str:
    total = max(result.duration_s, 1e-6)
    w = 1000
    x = lambda t: t / total * w  # noqa: E731
    parts = [f'<svg viewBox="0 0 {w} 46" class="timeline" role="img" aria-label="Recording timeline">']
    parts.append(f'<rect x="0" y="14" width="{w}" height="14" class="tl-base"/>')
    # Non-informative frames, drawn as thin ticks so short dropouts are still visible.
    tl = result.timeline
    for i, (t, ok) in enumerate(tl):
        if not ok:
            nxt = tl[i + 1][0] if i + 1 < len(tl) else t + 0.2
            parts.append(f'<rect x="{x(t):.1f}" y="14" width="{max(x(nxt) - x(t), 0.6):.1f}" height="14" class="tl-poor"/>')
    for b in result.blind_segments:
        parts.append(
            f'<rect x="{x(b.start_s):.1f}" y="10" width="{max(x(b.end_s) - x(b.start_s), 1):.1f}" height="22" class="tl-blind">'
            f"<title>Blind segment {_fmt_time(b.start_s)}-{_fmt_time(b.end_s)}: {html.escape(b.reason)}</title></rect>"
        )
    for f in result.findings:
        cls = "tl-missed" if f.status == FindingStatus.POTENTIALLY_MISSED else "tl-finding"
        parts.append(
            f'<a href="#{f.finding_id}"><rect x="{x(f.first.timestamp_s):.1f}" y="4" '
            f'width="{max(x(f.last.timestamp_s) - x(f.first.timestamp_s), 3):.1f}" height="34" rx="2" class="{cls}">'
            f"<title>{f.finding_id} {_fmt_time(f.first.timestamp_s)}</title></rect></a>"
        )
    parts.append(f'<text x="0" y="45" class="tl-label">00:00</text>')
    parts.append(f'<text x="{w}" y="45" text-anchor="end" class="tl-label">{_fmt_time(total)}</text>')
    parts.append("</svg>")
    return "".join(parts)


def _finding_card(result: AnalysisResult, f) -> str:
    imgs = result.images[f.finding_id]
    c = f.characterisation
    if c.model:
        char = (
            f"<p><b>Characterisation ({html.escape(c.model)}):</b> {html.escape(str(c.histology_prediction))}, "
            f"malignancy risk {c.malignancy_risk:.2f}</p>"
        )
    else:
        char = f'<p class="muted">{html.escape(c.note)}</p>'
    matched = f" &rarr; report {html.escape(f.matched_report_id)}" if f.matched_report_id else ""
    reasons = "".join(f"<li>{html.escape(r)}</li>" for r in f.rationale)
    return f"""
<section class="card" id="{f.finding_id}" data-finding="{f.finding_id}">
  <header>
    <h3>{f.finding_id} <span class="time">{_fmt_time(f.best.timestamp_s)}</span></h3>
    <span class="badge {f.status.value}">{STATUS_LABEL[f.status]}{matched}</span>
  </header>
  <div class="views">
    <figure>{_img_tag(imgs['overlay'], f'{f.finding_id} evidence heatmap')}<figcaption>Evidence heatmap + outline</figcaption></figure>
    <figure>{_img_tag(imgs['original'], f'{f.finding_id} original frame')}<figcaption>Original frame #{f.best.frame_index}</figcaption></figure>
  </div>
  <figure class="strip">{_img_tag(imgs['filmstrip'], f'{f.finding_id} frames over time')}<figcaption>Appearance across the {f.duration_s:.1f} s it was in view</figcaption></figure>
  <h4>Why this was flagged</h4>
  <ul>{reasons}</ul>
  {char}
  <fieldset class="decision">
    <legend>Second reader decision</legend>
    <label><input type="radio" name="d-{f.finding_id}" value="true_lesion"> Lesion confirmed</label>
    <label><input type="radio" name="d-{f.finding_id}" value="false_positive"> Not a lesion</label>
    <label><input type="radio" name="d-{f.finding_id}" value="uncertain"> Uncertain / needs MDT</label>
    <textarea name="c-{f.finding_id}" placeholder="Comment (location, size, Paris class, action)"></textarea>
  </fieldset>
</section>"""


CSS = """
:root{--bg:#f7f7f5;--fg:#1d1d1b;--muted:#6b6b66;--card:#fff;--line:#deddd8;--accent:#0f6e8c;
--missed:#b4321f;--ok:#2e7d4f;--neutral:#7a6a1e;--poor:#c9c6bd;--blind:#e7a33a}
@media (prefers-color-scheme:dark){:root{--bg:#161615;--fg:#ecebe6;--muted:#a3a29b;--card:#212120;--line:#383835;
--accent:#53b5d4;--missed:#ef7b67;--ok:#6cc690;--neutral:#d9c46b;--poor:#4a4945;--blind:#c98a2a}}
*{box-sizing:border-box}body{margin:0;background:var(--bg);color:var(--fg);font:15px/1.5 system-ui,-apple-system,Segoe UI,sans-serif}
main{max-width:1100px;margin:0 auto;padding:24px 16px 64px}h1{margin:0 0 4px;font-size:1.6rem}
.notice{border-left:4px solid var(--missed);background:var(--card);padding:10px 14px;margin:16px 0}
.muted{color:var(--muted)}.tiles{display:grid;grid-template-columns:repeat(auto-fit,minmax(150px,1fr));gap:10px;margin:18px 0}
.tile{background:var(--card);border:1px solid var(--line);border-radius:8px;padding:10px 12px}
.tile b{display:block;font-size:1.5rem;font-variant-numeric:tabular-nums}.tile.alert b{color:var(--missed)}
.timeline{width:100%;height:auto;margin:6px 0 2px}.tl-base{fill:var(--ok);opacity:.35}.tl-poor{fill:var(--poor)}
.tl-blind{fill:var(--blind);opacity:.75}.tl-finding{fill:var(--accent)}.tl-missed{fill:var(--missed)}.tl-label{fill:var(--muted);font-size:11px}
.legend span{margin-right:14px;font-size:.85rem}.legend i{display:inline-block;width:12px;height:12px;border-radius:2px;margin-right:4px;vertical-align:-1px}
.card{background:var(--card);border:1px solid var(--line);border-radius:10px;padding:16px;margin:18px 0}
.card header{display:flex;justify-content:space-between;align-items:center;flex-wrap:wrap;gap:8px}.card h3{margin:0}
.time{color:var(--muted);font-weight:400;font-variant-numeric:tabular-nums}
.badge{padding:3px 10px;border-radius:999px;font-size:.85rem;font-weight:600;color:#fff}
.badge.potentially_missed{background:var(--missed)}.badge.reported{background:var(--ok)}.badge.unreviewed{background:var(--neutral)}
.views{display:grid;grid-template-columns:repeat(auto-fit,minmax(260px,1fr));gap:12px;margin-top:12px}
figure{margin:0}figure img{width:100%;border-radius:6px;display:block}figcaption{font-size:.8rem;color:var(--muted)}
.strip{margin-top:10px;overflow-x:auto}.strip img{width:auto;max-width:none;height:110px}
.decision{border:1px solid var(--line);border-radius:8px;margin-top:10px}.decision label{margin-right:16px;white-space:nowrap}
textarea{width:100%;margin-top:8px;min-height:48px;background:var(--bg);color:var(--fg);border:1px solid var(--line);border-radius:6px;padding:6px}
table{border-collapse:collapse;width:100%;background:var(--card)}td,th{border:1px solid var(--line);padding:6px 8px;text-align:left}
button{background:var(--accent);color:#fff;border:0;border-radius:6px;padding:10px 16px;font-size:1rem;cursor:pointer}
input[type=text]{background:var(--bg);color:var(--fg);border:1px solid var(--line);border-radius:6px;padding:8px}
"""

JS = """
function exportDecisions(){
  const out={report_generated_utc:REPORT_META.generated_utc,input:REPORT_META.input,
    reviewer:document.getElementById('reviewer').value,exported_utc:new Date().toISOString(),decisions:[]};
  document.querySelectorAll('[data-finding]').forEach(c=>{
    const id=c.dataset.finding, d=c.querySelector('input[type=radio]:checked');
    out.decisions.push({finding_id:id,decision:d?d.value:null,comment:c.querySelector('textarea').value});
  });
  const a=document.createElement('a');
  a.href=URL.createObjectURL(new Blob([JSON.stringify(out,null,2)],{type:'application/json'}));
  a.download='review_decisions.json';a.click();
}
"""


def write_html(result: AnalysisResult, data: dict, path: Path) -> None:
    s = data["summary"]
    informative_pct = 100 * s["frames_informative"] / max(s["frames_analysed"], 1)
    missed_tile = (
        f'<div class="tile alert"><b>{s["potentially_missed"]}</b>potentially missed</div>'
        if s["report_supplied"]
        else '<div class="tile"><b>&ndash;</b>no procedure report supplied</div>'
    )
    cards = "".join(_finding_card(result, f) for f in result.findings) or "<p>No findings above threshold.</p>"
    blind_rows = "".join(
        f"<tr><td>{_fmt_time(b.start_s)}</td><td>{_fmt_time(b.end_s)}</td><td>{b.duration_s:.1f} s</td><td>{html.escape(b.reason)}</td></tr>"
        for b in result.blind_segments
    )
    unmatched_rows = "".join(
        f"<tr><td>{html.escape(r.report_id)}</td><td>{_fmt_time(r.time_s) if r.time_s is not None else 'frame ' + str(r.frame_index)}</td>"
        f"<td>{html.escape(r.location or '')}</td><td>{html.escape(r.note or '')}</td></tr>"
        for r in result.unmatched_reported
    )
    meta = json.dumps({"generated_utc": data["generated_utc"], "input": data["input"]})
    doc = f"""<!doctype html><html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1"><title>SecondLook Review</title><style>{CSS}</style></head>
<body><main>
<h1>Second-look review</h1>
<div class="muted">{html.escape(Path(result.input_path).name)} &middot; {result.config.modality.value} &middot;
detector {html.escape(result.detector)} v{html.escape(result.detector_version)} &middot; SecondLook-CRC v{__version__} &middot; {data["generated_utc"][:19]} UTC</div>
<div class="notice">{html.escape(INTENDED_USE_NOTICE)}</div>
<div class="tiles">
  <div class="tile"><b>{_fmt_time(s["duration_s"])}</b>recording length</div>
  <div class="tile"><b>{s["findings"]}</b>findings flagged</div>
  {missed_tile}
  <div class="tile"><b>{informative_pct:.0f}%</b>frames of reviewable quality</div>
  <div class="tile"><b>{s["blind_segments"]}</b>blind segments ({s["blind_time_s"]:.0f} s)</div>
</div>
<h2>Timeline</h2>
{_timeline_svg(result)}
<div class="legend"><span><i style="background:var(--missed)"></i>potentially missed</span>
<span><i style="background:var(--accent)"></i>other finding</span><span><i style="background:var(--blind)"></i>blind segment</span>
<span><i style="background:var(--poor)"></i>poor-quality frame</span></div>
<h2>Findings</h2>
{cards}
<h2>Blind segments</h2>
<p class="muted">Stretches where image quality was too poor to review. Lesions here can be missed by both the endoscopist and the AI.</p>
{f"<table><tr><th>From</th><th>To</th><th>Length</th><th>Reason</th></tr>{blind_rows}</table>" if blind_rows else "<p>None.</p>"}
{f"<h2>Reported lesions not found by the AI</h2><table><tr><th>ID</th><th>When</th><th>Location</th><th>Note</th></tr>{unmatched_rows}</table>" if unmatched_rows else ""}
<h2>Sign-off</h2>
<p><input type="text" id="reviewer" placeholder="Reviewer name / GMC number"> <button onclick="exportDecisions()">Export review decisions</button></p>
</main><script>const REPORT_META={meta};{JS}</script></body></html>"""
    path.write_text(doc, encoding="utf-8")


def write(result: AnalysisResult, out_dir: Path | str) -> dict:
    out_dir = Path(out_dir)
    (out_dir / "findings").mkdir(parents=True, exist_ok=True)
    data = result_dict(result)
    (out_dir / "result.json").write_text(json.dumps(data, indent=2))
    for fid, imgs in result.images.items():
        for kind, img in imgs.items():
            cv2.imwrite(str(out_dir / "findings" / f"{fid}_{kind}.png"), img)
    write_html(result, data, out_dir / "report.html")
    return data
