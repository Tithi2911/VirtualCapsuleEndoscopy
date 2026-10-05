"""Write analysis results as machine-readable JSON plus a self-contained HTML report.

The HTML report is a single file (images embedded) so it can be attached to a
patient record, shared inside an MDT, or archived for audit without a server.
It includes reviewer controls (confirm / reject / unsure + comment per finding)
that export a decisions JSON: this is the second-reader sign-off and, with
consent and governance, the labelled feedback used to retrain the model.

When a lesion classifier is configured, each finding also shows the AI optical
diagnosis: benign, precancerous or suspected cancer, or indeterminate when the
model abstained, with probabilities as numbers (not colour alone), labelled as
calibrated only when the model's temperature was actually fitted, the frames it
used and an explanation map. The classifier's file hash, training data,
validation status and intended use are shown too, so a reader can tell a
synthetic test model from a validated one. Optical diagnosis predicts histology
from the endoscopic image, so the safety statement sits next to every
prediction. The reviewer records their own optical diagnosis and, later, the
histology result; exported with the AI output, these become labelled data for
auditing and retraining the classifier.

Model names and versions come from a sidecar file and are treated as untrusted:
every piece of text is HTML-escaped, and JSON embedded in the page is escaped so
it cannot close the script element.
"""

from __future__ import annotations

import base64
import html
import json
import math
from datetime import datetime, timezone
from pathlib import Path

import cv2

from . import INTENDED_USE_NOTICE, __version__
from .characterise import NullCharacteriser
from .diagnosis.taxonomy import CATEGORIES, DISPLAY_NAME, DiagnosticCategory, category_for
from .models import Characterisation, Finding, FindingStatus, to_jsonable
from .pipeline import AnalysisResult

STATUS_LABEL = {
    FindingStatus.POTENTIALLY_MISSED: "Potentially missed",
    FindingStatus.REPORTED: "Matches report",
    FindingStatus.UNREVIEWED: "Not compared",
}

# AI category of a finding: one of CATEGORIES, or one of these two.
INDETERMINATE = "indeterminate"  # a classifier ran but abstained
UNCHARACTERISED = "uncharacterised"  # no classifier ran
AI_CATEGORY_KEYS = [*CATEGORIES, INDETERMINATE, UNCHARACTERISED]

CATEGORY_LABEL = {
    **DISPLAY_NAME,
    INDETERMINATE: "Indeterminate (AI abstained)",
    UNCHARACTERISED: "Not characterised",
}
# Badge for INDETERMINATE findings of a recording the classifier was not applied to at all.
NOT_APPLIED_LABEL = "Not applied (no AI category)"

AI_DIAGNOSIS_NOTICE = (
    "AI optical diagnosis predicts histology from the endoscopic image; it is not a histological "
    "diagnosis and must not be used alone for resect-and-discard or diagnose-and-leave decisions. "
    "Confirm with histopathology. The AI gives a category to every detection, including ones that are "
    "not lesions: first decide whether the finding is a lesion at all."
)

# Example shown in the histology field; exported histology becomes training labels, so it must be
# a label train_classifier.py accepts (taxonomy.category_for).
HISTOLOGY_EXAMPLE = "tubular adenoma, low-grade dysplasia"

REVIEWER_CATEGORY_OPTIONS = [
    ("", "Not assessed"),
    *((c, DISPLAY_NAME[c]) for c in CATEGORIES),
    ("unsure", "Unsure"),
]

esc = html.escape


def _fmt_time(t: float) -> str:
    m, s = divmod(t, 60)
    return f"{int(m):02d}:{s:04.1f}"


def _finite(v) -> bool:
    try:
        return math.isfinite(float(v))
    except (TypeError, ValueError):
        return False


def _pct(p: float) -> str:
    """Whole-number percentage that never rounds a small chance to 0% or a large one to 100%."""
    if not _finite(p):
        return "n/a"
    v = 100 * float(p)
    if 0 < v < 1:
        return "<1%"
    if 99 < v < 100:
        return ">99%"
    return f"{v:.0f}%"


def _numbers_ok(c: Characterisation) -> bool:
    """False if a characteriser returned NaN or infinite numbers, which must never read as a category."""
    values = [*c.probabilities.values(), c.confidence, c.frame_agreement, c.malignancy_risk,
              c.peak_malignancy_risk, c.neoplasia_risk]
    return all(v is None or _finite(v) for v in values)


def _benign_supported(c: Characterisation) -> bool:
    """The deployed rule (metrics.decide) for any characteriser, not only the bundled one: a lesion is
    called benign only when P(benign) > P(precancerous) + P(cancerous), which needs all three numbers."""
    p = c.probabilities or {}
    if not all(k in p and _finite(p[k]) for k in CATEGORIES):
        return False
    return float(p["benign"]) > float(p["precancerous"]) + float(p["cancerous"])


def ai_category(c: Characterisation) -> str:
    """The finding's AI category, INDETERMINATE if the model abstained, or UNCHARACTERISED if none ran.
    A benign category that its own probabilities do not support is INDETERMINATE too."""
    if not c.model:
        return UNCHARACTERISED
    if c.abstained or not c.category or not _numbers_ok(c):
        return INDETERMINATE
    try:
        category = category_for(c.category).value
    except KeyError:
        return INDETERMINATE
    if category == DiagnosticCategory.BENIGN.value and not _benign_supported(c):
        return INDETERMINATE
    return category


def _abstain_reason(c: Characterisation) -> str:
    if c.abstain_reason and (c.abstained or not c.category):
        return c.abstain_reason
    if not _numbers_ok(c):
        return "the AI output contains values that are not numbers (NaN or infinity)"
    if c.category and not c.abstained:
        try:
            benign = category_for(c.category) == DiagnosticCategory.BENIGN
        except KeyError:
            return f"unrecognised category {c.category!r}"
        if benign and not _benign_supported(c):
            p = c.probabilities or {}
            if not all(k in p for k in CATEGORIES):
                return "a benign category was returned without the probabilities needed to check it"
            return (f"not called benign because P(precancerous) + P(cancerous) = "
                    f"{float(p['precancerous']) + float(p['cancerous']):.2f} is not below P(benign) = "
                    f"{float(p['benign']):.2f}")
    return c.abstain_reason or "no category returned"


def _characterisation_json(c: Characterisation, cat: str) -> dict:
    """The characterisation as stored in result.json. When the report withholds the category, so does
    the JSON (category None, abstained), so no consumer reads a category the report did not show; the
    characteriser's own value is kept as `withheld_category` for audit."""
    d = to_jsonable(c)
    if cat == INDETERMINATE and (d.get("category") or not d.get("abstained")):
        d.update(withheld_category=d.get("category"), category=None, abstained=True, abstain_reason=_abstain_reason(c))
    return d


def classifier_not_applied(result: AnalysisResult) -> bool:
    """True when the classifier refused this recording (modality or working size), so no finding was classified."""
    return bool((result.characteriser_info or {}).get("unsupported_reason"))


def _strict(obj):
    """NaN and infinity -> None, so result.json is strict JSON (capsule analysis_fps is infinite)."""
    if isinstance(obj, float):
        return obj if math.isfinite(obj) else None
    if isinstance(obj, dict):
        return {k: _strict(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_strict(v) for v in obj]
    return obj


def classifier_used(result: AnalysisResult) -> bool:
    return (result.characteriser or NullCharacteriser.name) != NullCharacteriser.name or any(
        f.characterisation.model for f in result.findings
    )


def result_dict(result: AnalysisResult) -> dict:
    categories = [ai_category(f.characterisation) for f in result.findings]
    return _strict({
        "software": {"name": "SecondLook-CRC", "version": __version__},
        "intended_use_notice": INTENDED_USE_NOTICE,
        "ai_diagnosis_notice": AI_DIAGNOSIS_NOTICE if classifier_used(result) else None,
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "input": result.input_path,
        "modality": result.config.modality.value,
        "config": to_jsonable(result.config),
        "detector": {"name": result.detector, "version": result.detector_version},
        "characteriser": {"name": result.characteriser, "version": result.characteriser_version,
                          **to_jsonable(result.characteriser_info)},
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
            "ai_categories": {k: categories.count(k) for k in AI_CATEGORY_KEYS},
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
                "ai_category": cat,
                "characterisation": _characterisation_json(f.characterisation, cat),
            }
            for f, cat in zip(result.findings, categories)
        ],
        "blind_segments": to_jsonable(result.blind_segments),
        "reported_not_found_by_ai": to_jsonable(result.unmatched_reported),
    })


def _img_tag(image, alt: str, cls: str = "") -> str:
    ok, buf = cv2.imencode(".jpg", image, [cv2.IMWRITE_JPEG_QUALITY, 88])
    data = base64.b64encode(buf.tobytes()).decode() if ok else ""
    return f'<img class="{esc(cls)}" alt="{esc(alt)}" src="data:image/jpeg;base64,{data}">'


def _script_json(obj) -> str:
    """JSON safe to embed in a <script> element: no '</script>' or '<!--' can appear in it."""
    return json.dumps(obj).replace("<", "\\u003c").replace(">", "\\u003e").replace("&", "\\u0026")


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
            f"<title>Blind segment {_fmt_time(b.start_s)}-{_fmt_time(b.end_s)}: {esc(b.reason)}</title></rect>"
        )
    for f in result.findings:
        cls = "tl-missed" if f.status == FindingStatus.POTENTIALLY_MISSED else "tl-finding"
        fid = esc(f.finding_id)
        parts.append(
            f'<a href="#{fid}"><rect x="{x(f.first.timestamp_s):.1f}" y="4" '
            f'width="{max(x(f.last.timestamp_s) - x(f.first.timestamp_s), 3):.1f}" height="34" rx="2" class="{cls}">'
            f"<title>{fid} {_fmt_time(f.first.timestamp_s)}</title></rect></a>"
        )
    parts.append('<text x="0" y="45" class="tl-label">00:00</text>')
    parts.append(f'<text x="{w}" y="45" text-anchor="end" class="tl-label">{_fmt_time(total)}</text>')
    parts.append("</svg>")
    return "".join(parts)


def _category_badge(cat: str, not_applied: bool = False) -> str:
    label = NOT_APPLIED_LABEL if not_applied and cat == INDETERMINATE else CATEGORY_LABEL[cat]
    return f'<span class="cat {cat}">{esc(label)}</span>'


def _probability_table(f: Finding, cat: str) -> str:
    c = f.characterisation
    rows = []
    for k in CATEGORIES:
        if k not in c.probabilities:
            continue
        p = float(c.probabilities[k]) if _finite(c.probabilities[k]) else 0.0
        top = ' class="top"' if k == cat else ""
        rows.append(
            f'<tr{top}><th scope="row">{esc(DISPLAY_NAME[k])}</th>'
            f'<td><span class="bar" aria-hidden="true"><span class="fill {k}" style="width:{min(max(100 * p, 0), 100):.1f}%"></span></span></td>'
            f'<td class="pct">{esc(_pct(c.probabilities[k]))}</td></tr>'
        )
    if not rows:
        return ""
    over = f", averaged over {c.frames_used} frame{'s' if c.frames_used != 1 else ''}" if c.frames_used else ""
    label = f"AI category probabilities for {f.finding_id}"
    caption = (f"Probability per category{over} (calibrated on the model's own validation data; a model output, "
               "not this patient's risk)" if c.calibrated is True else
               f"Model probability per category{over} (not calibrated: read as a ranking, not as a risk)")
    return (
        f'<table class="probs" aria-label="{esc(label)}"><caption>{esc(caption)}</caption>'
        f'{"".join(rows)}</table>'
    )


def _diagnosis_facts(c: Characterisation, cat: str) -> str:
    facts = []
    if c.confidence is not None:
        facts.append(("Confidence" if cat != INDETERMINATE else "Highest probability", _pct(c.confidence)))
    facts.append(("Frames used", str(c.frames_used)))
    if c.frame_agreement is not None:
        n = c.frames_used
        agree = _pct(c.frame_agreement)
        facts.append(("Frame agreement", f"{agree} ({round(c.frame_agreement * n)} of {n} frames)" if n else agree))
    if c.neoplasia_risk is not None:
        facts.append(("AI P(neoplastic: precancerous + cancer)", _pct(c.neoplasia_risk)))
    if c.peak_malignancy_risk is not None:
        facts.append(("Highest single-frame P(suspected cancer)", _pct(c.peak_malignancy_risk)))
    if c.histology_prediction:
        facts.append(("Predicted subtype", c.histology_prediction))
    if c.morphology:
        facts.append(("Morphology", c.morphology))
    facts.append(("Model", f"{c.model} v{c.model_version or 'unversioned'}"))
    items = "".join(f"<div><dt>{esc(k)}</dt><dd>{esc(str(v))}</dd></div>" for k, v in facts)
    return f'<dl class="dx-facts">{items}</dl>'


def _short_hash(value) -> str:
    return str(value)[:12] if value else ""


def _model_line(result: AnalysisResult, c: Characterisation) -> str:
    """Which model and how far it has been validated, next to every AI category."""
    info = result.characteriser_info or {}
    if c.model != result.characteriser or not info:
        return ""
    sha = f" (SHA-256 {esc(_short_hash(info.get('sha256')))})" if info.get("sha256") else ""
    status = esc(str(info.get("validation_status", "")))
    return (f'<p class="dx-model">Model {esc(c.model)} v{esc(c.model_version or "unversioned")}{sha}. {status} '
            'Research use only; see the classifier details at the top of the report.</p>')


def _diagnosis_block(result: AnalysisResult, f: Finding) -> str:
    c = f.characterisation
    if not c.model:
        return f'<p class="muted">{esc(c.note)}</p>'
    fid = esc(f.finding_id)
    cat = ai_category(c)
    lead = ""
    if cat == INDETERMINATE:
        lead = (
            f'<p class="abstain"><b>No AI category given:</b> {esc(_abstain_reason(c))}. '
            "This lesion needs assessment by the reviewer.</p>"
        )
    explanation = result.images.get(f.finding_id, {}).get("diagnosis")
    figure = ""
    if explanation is not None:
        figure = (
            f'<figure class="dx-explain">{_img_tag(explanation, f"{f.finding_id} AI optical diagnosis explanation map")}'
            "<figcaption>Image regions that most influenced the AI category (occlusion sensitivity: brighter means "
            "hiding that region made the AI less sure of its category). Brightness away from the lesion is a "
            "warning sign. An endoscopic image, not a pathology image.</figcaption></figure>"
        )
    return f"""
  <section class="dx" aria-labelledby="dx-{fid}">
    <h4 id="dx-{fid}">AI optical diagnosis</h4>
    <p class="dx-cat">{_category_badge(cat, classifier_not_applied(result))}</p>
    {lead}
    <div class="dx-body">
      <div>{_probability_table(f, cat)}{_diagnosis_facts(c, cat)}</div>
      {figure}
    </div>
    {_model_line(result, c)}
    <p class="safety">{esc(AI_DIAGNOSIS_NOTICE)}</p>
  </section>"""


def _finding_card(result: AnalysisResult, f: Finding) -> str:
    imgs = result.images[f.finding_id]
    fid = esc(f.finding_id)
    matched = f" &rarr; report {esc(f.matched_report_id)}" if f.matched_report_id else ""
    reasons = "".join(f"<li>{esc(r)}</li>" for r in f.rationale)
    options = "".join(f'<option value="{v}">{esc(t)}</option>' for v, t in REVIEWER_CATEGORY_OPTIONS)
    return f"""
<section class="card" id="{fid}" data-finding="{fid}">
  <header>
    <h3>{fid} <span class="time">{_fmt_time(f.best.timestamp_s)}</span></h3>
    <span class="badge {f.status.value}">{STATUS_LABEL[f.status]}{matched}</span>
  </header>
  <div class="views">
    <figure>{_img_tag(imgs['overlay'], f'{f.finding_id} evidence heatmap')}<figcaption>Evidence heatmap + outline</figcaption></figure>
    <figure>{_img_tag(imgs['original'], f'{f.finding_id} original frame')}<figcaption>Original frame #{f.best.frame_index}</figcaption></figure>
  </div>
  <figure class="strip">{_img_tag(imgs['filmstrip'], f'{f.finding_id} frames over time')}<figcaption>Appearance across the {f.duration_s:.1f} s it was in view</figcaption></figure>
  <h4>Why this was flagged</h4>
  <ul>{reasons}</ul>
  {_diagnosis_block(result, f)}
  <fieldset class="decision">
    <legend>Second reader decision</legend>
    <label><input type="radio" name="d-{fid}" value="true_lesion"> Lesion confirmed</label>
    <label><input type="radio" name="d-{fid}" value="false_positive"> Not a lesion</label>
    <label><input type="radio" name="d-{fid}" value="uncertain"> Uncertain / needs MDT</label>
    <div class="fields">
      <label class="field">Reviewer optical diagnosis
        <select name="rc-{fid}" class="reviewer-category">{options}</select></label>
      <label class="field">Histology result (when available)
        <input type="text" name="h-{fid}" class="histology" autocomplete="off" placeholder="e.g. {esc(HISTOLOGY_EXAMPLE)}"></label>
    </div>
    <textarea name="c-{fid}" placeholder="Comment (location, size, Paris class, action)"></textarea>
  </fieldset>
</section>"""


# Priority groups: suspected cancer, then everything without an AI category (it needs a human
# decision and is never treated as benign), then precancerous, then benign.
PRIORITY_GROUP = {DiagnosticCategory.CANCEROUS.value: 0, INDETERMINATE: 1, UNCHARACTERISED: 1,
                  DiagnosticCategory.PRECANCEROUS.value: 2, DiagnosticCategory.BENIGN.value: 3}


def _priority_key(f: Finding) -> tuple:
    c = f.characterisation
    known = _finite(c.malignancy_risk)
    return (
        PRIORITY_GROUP[ai_category(c)],
        0 if known else 1,
        -float(c.malignancy_risk) if known else 0.0,
        -float(c.neoplasia_risk) if _finite(c.neoplasia_risk) else 0.0,
    )


def _priority_list(result: AnalysisResult) -> str:
    """Suspected cancer first, then findings without an AI category, then precancerous, then benign;
    within each group by the AI probability of the suspected-cancer category."""
    items = []
    for f in sorted(result.findings, key=_priority_key):
        c = f.characterisation
        cat = ai_category(c)
        fid = esc(f.finding_id)
        if _finite(c.malignancy_risk):
            parts = [f"AI P(suspected cancer) {esc(_pct(c.malignancy_risk))}"]
            if _finite(c.neoplasia_risk):
                parts.append(f"P(neoplastic) {esc(_pct(c.neoplasia_risk))}")
            if _finite(c.peak_malignancy_risk) and c.peak_malignancy_risk - c.malignancy_risk >= 0.1:
                parts.append(f"single-frame peak P(suspected cancer) {esc(_pct(c.peak_malignancy_risk))}")
        else:
            parts = ["no AI estimate"]
        if cat in (INDETERMINATE, UNCHARACTERISED):
            parts.append("<b>needs human review</b>")
        if f.status == FindingStatus.POTENTIALLY_MISSED:
            parts.append("<b>potentially missed</b>")
        items.append(
            f'<li><a href="#{fid}">{fid}</a> <span class="time">{_fmt_time(f.best.timestamp_s)}</span> '
            f'{_category_badge(cat, classifier_not_applied(result))} <span class="risk">{" &middot; ".join(parts)}</span></li>'
        )
    return f"""<h2>Priority list</h2>
<p class="muted">Suspected cancer first, then findings without an AI category (they need human review and are
never treated as benign), then precancerous, then benign; within each group by the AI probability of the
suspected-cancer category. These are model outputs, not clinical risks. Calibration on the model's own
validation data does not make them risks for your patients.</p>
<ol class="priority">{"".join(items)}</ol>"""


def _classifier_section(result: AnalysisResult) -> str:
    """The lesion classifier's provenance, calibration, validation status and intended use."""
    info = result.characteriser_info or {}
    rows = [("Model", f"{result.characteriser} v{result.characteriser_version}")]
    if info.get("sha256"):
        rows.append(("Model file SHA-256", str(info["sha256"])))
    if info.get("sidecar_sha256"):
        rows.append(("Sidecar SHA-256", str(info["sidecar_sha256"])))
    if info.get("modalities"):
        rows.append(("Trained for", ", ".join(map(str, info["modalities"]))))
    if "calibrated" in info:
        rows.append(("Probabilities", "calibrated on the model's own validation data (temperature scaling); model "
                     "outputs, not risks for your patients" if info["calibrated"] else "not calibrated"))
    if info.get("training_data"):
        rows.append(("Training data", ", ".join(map(str, info["training_data"]))))
    if info.get("init_training_data"):
        rows.append(("Fine-tuned from a model trained on", ", ".join(map(str, info["init_training_data"]))))
    if info.get("validation_status"):
        rows.append(("Validation status", str(info["validation_status"])))
    if info.get("intended_use"):
        rows.append(("Intended use", str(info["intended_use"])))
    # Long free-text fields take a full row so they are not squeezed into a narrow column.
    wide = {"Validation status", "Intended use"}
    items = "".join(
        f'<div{" class=wide" if k in wide else ""}><dt>{esc(k)}</dt><dd>{esc(v)}</dd></div>' for k, v in rows
    )
    warnings = "".join(f"<li>{esc(str(w))}</li>" for w in info.get("validation_warnings") or [])
    warn_html = f'<p>Warnings recorded when the model was trained:</p><ul>{warnings}</ul>' if warnings else ""
    unsupported = ""
    if info.get("unsupported_reason"):
        unsupported = (f'<p class="abstain"><b>Not applied to this recording:</b> '
                       f'{esc(str(info["unsupported_reason"]))}. Every finding is left without an AI category.</p>')
    return f"""<section class="classifier" aria-labelledby="clf-h"><h2 id="clf-h">AI lesion classifier</h2>
{unsupported}<dl class="dx-facts">{items}</dl>{warn_html}</section>"""


CSS = """
:root{--bg:#f7f7f5;--fg:#1d1d1b;--muted:#6b6b66;--card:#fff;--line:#deddd8;--accent:#0f6e8c;
--missed:#b4321f;--ok:#2e7d4f;--neutral:#7a6a1e;--poor:#c9c6bd;--blind:#e7a33a;
--benign:#1d6b52;--precancer:#8a4b00;--cancer:#a3123a;--indeterminate:#555a66;--on-cat:#fff}
@media (prefers-color-scheme:dark){:root{--bg:#161615;--fg:#ecebe6;--muted:#a3a29b;--card:#212120;--line:#383835;
--accent:#53b5d4;--missed:#ef7b67;--ok:#6cc690;--neutral:#d9c46b;--poor:#4a4945;--blind:#c98a2a;
--benign:#74d1b0;--precancer:#f2b35e;--cancer:#f58ba0;--indeterminate:#b9bdc8;--on-cat:#161615}}
*{box-sizing:border-box}body{margin:0;background:var(--bg);color:var(--fg);font:15px/1.5 system-ui,-apple-system,Segoe UI,sans-serif}
main{max-width:1100px;margin:0 auto;padding:24px 16px 64px}h1{margin:0 0 4px;font-size:1.6rem}a{color:var(--accent)}
.notice{border-left:4px solid var(--missed);background:var(--card);padding:10px 14px;margin:16px 0}
.muted{color:var(--muted)}.tiles{display:grid;grid-template-columns:repeat(auto-fit,minmax(min(100%,128px),1fr));gap:10px;margin:18px 0}
.tile{background:var(--card);border:1px solid var(--line);border-radius:8px;padding:10px 12px}
.tile b{display:block;font-size:1.5rem;font-variant-numeric:tabular-nums}.tile.alert b{color:var(--missed)}
.timeline{width:100%;height:auto;margin:6px 0 2px}.tl-base{fill:var(--ok);opacity:.35}.tl-poor{fill:var(--poor)}
.tl-blind{fill:var(--blind);opacity:.75}.tl-finding{fill:var(--accent)}.tl-missed{fill:var(--missed)}.tl-label{fill:var(--muted);font-size:11px}
.legend span{margin-right:14px;font-size:.85rem}.legend i{display:inline-block;width:12px;height:12px;border-radius:2px;margin-right:4px;vertical-align:-1px}
.card{background:var(--card);border:1px solid var(--line);border-radius:10px;padding:16px;margin:18px 0}
.card header{display:flex;justify-content:space-between;align-items:center;flex-wrap:wrap;gap:8px}.card h3{margin:0}
.time{color:var(--muted);font-weight:400;font-variant-numeric:tabular-nums}
.badge{padding:3px 10px;border-radius:999px;font-size:.85rem;font-weight:600;color:var(--on-cat)}
.badge.potentially_missed{background:var(--missed)}.badge.reported{background:var(--ok)}.badge.unreviewed{background:var(--neutral)}
.views{display:grid;grid-template-columns:repeat(auto-fit,minmax(min(100%,260px),1fr));gap:12px;margin-top:12px}
figure{margin:0}figure img{width:100%;border-radius:6px;display:block}figcaption{font-size:.8rem;color:var(--muted)}
.strip{margin-top:10px;overflow-x:auto}.strip img{width:auto;max-width:none;height:110px}
.cat{display:inline-block;padding:2px 9px;border-radius:5px;font-size:.85rem;font-weight:700;line-height:1.4;color:var(--on-cat);background:var(--indeterminate)}
.cat.benign{background:var(--benign)}.cat.precancerous{background:var(--precancer)}.cat.cancerous{background:var(--cancer)}
.cat.uncharacterised{background:transparent;color:var(--muted);border:1px solid var(--line);font-weight:600}
.dx{border-top:1px solid var(--line);margin-top:14px}.dx h4{margin:12px 0 6px}.dx-cat{margin:0 0 6px}.dx-cat .cat{font-size:1rem;padding:4px 12px}
.abstain{margin:6px 0}
.dx-body{display:grid;grid-template-columns:repeat(auto-fit,minmax(min(100%,300px),1fr));gap:14px;align-items:start;margin-top:8px}
table.probs{background:transparent;table-layout:fixed;width:100%}.probs caption{text-align:left;font-size:.8rem;color:var(--muted);padding-bottom:4px}
.probs th,.probs td{border:0;padding:3px 8px 3px 0;vertical-align:middle}.probs th{font-weight:400;width:46%;overflow-wrap:anywhere}
.probs td.pct{width:3.8em;padding-right:0;text-align:right;font-variant-numeric:tabular-nums}.probs tr.top th,.probs tr.top td.pct{font-weight:700}
.bar{display:block;height:12px;background:var(--line);border-radius:3px;overflow:hidden}.bar .fill{display:block;height:100%;background:var(--indeterminate)}
.fill.benign{background:var(--benign)}.fill.precancerous{background:var(--precancer)}.fill.cancerous{background:var(--cancer)}
.dx-facts{display:grid;grid-template-columns:repeat(auto-fit,minmax(min(100%,140px),1fr));gap:6px 14px;margin:12px 0 0}
.dx-facts dt{font-size:.8rem;color:var(--muted)}.dx-facts dd{margin:0;font-variant-numeric:tabular-nums;overflow-wrap:anywhere}
.dx-explain img{max-width:256px}.dx-explain figcaption{max-width:320px}
.dx-model{font-size:.85rem;color:var(--muted);margin:10px 0 0}
.classifier{background:var(--card);border:1px solid var(--line);border-radius:10px;padding:4px 16px 12px;margin:16px 0}
.classifier h2{font-size:1.1rem}.classifier dd{overflow-wrap:anywhere}.dx-facts .wide{grid-column:1/-1}
.safety{border-left:4px solid var(--missed);background:var(--bg);padding:8px 12px;margin:12px 0 0;font-size:.9rem}
.priority{padding-left:1.6em}.priority li{margin:6px 0;overflow-wrap:anywhere}.priority .risk{font-variant-numeric:tabular-nums}
.decision{border:1px solid var(--line);border-radius:8px;margin-top:10px;min-width:0}.decision label{margin-right:16px;white-space:nowrap}
.decision .fields{display:grid;grid-template-columns:repeat(auto-fit,minmax(min(100%,220px),1fr));gap:8px 16px;margin-top:10px}
.decision .field{display:block;margin:0;white-space:normal;font-size:.9rem}.field select,.field input{display:block;width:100%;margin-top:2px}
textarea{width:100%;margin-top:8px;min-height:48px;background:var(--bg);color:var(--fg);border:1px solid var(--line);border-radius:6px;padding:6px}
table{border-collapse:collapse;width:100%;background:var(--card)}td,th{border:1px solid var(--line);padding:6px 8px;text-align:left}
button{background:var(--accent);color:#fff;border:0;border-radius:6px;padding:10px 16px;font-size:1rem;cursor:pointer}
input[type=text],select{background:var(--bg);color:var(--fg);border:1px solid var(--line);border-radius:6px;padding:8px;font:inherit;max-width:100%}
"""

JS = """
function exportDecisions(){
  const out={report_generated_utc:REPORT_META.generated_utc,input:REPORT_META.input,
    characteriser:REPORT_META.characteriser,ai_diagnosis_notice:REPORT_META.ai_diagnosis_notice,
    reviewer:document.getElementById('reviewer').value,exported_utc:new Date().toISOString(),decisions:[]};
  document.querySelectorAll('[data-finding]').forEach(c=>{
    const id=c.dataset.finding, d=c.querySelector('input[type=radio]:checked'), ai=REPORT_META.findings[id]||{};
    const rc=c.querySelector('select.reviewer-category'), h=c.querySelector('input.histology');
    out.decisions.push({finding_id:id,decision:d?d.value:null,comment:c.querySelector('textarea').value,
      reviewer_category:rc&&rc.value?rc.value:null,histology:h&&h.value.trim()?h.value.trim():null,
      ai_category:ai.ai_category||null,ai_probabilities:ai.probabilities||{},ai_confidence:ai.confidence??null,
      ai_abstain_reason:ai.abstain_reason||null,ai_model:ai.model||null,ai_model_version:ai.model_version||null});
  });
  const a=document.createElement('a');
  a.href=URL.createObjectURL(new Blob([JSON.stringify(out,null,2)],{type:'application/json'}));
  a.download='review_decisions.json';a.click();
}
"""


def _report_meta(data: dict) -> dict:
    """What the export script needs: per-finding AI output, so exported decisions pair it with the reviewer's."""
    findings = {}
    for fd in data["findings"]:
        c = fd.get("characterisation") or {}
        findings[fd["id"]] = {
            "ai_category": fd["ai_category"],
            "probabilities": c.get("probabilities") or {},
            "confidence": c.get("confidence"),
            "abstain_reason": c.get("abstain_reason"),
            "model": c.get("model"),
            "model_version": c.get("model_version"),
        }
    return {
        "generated_utc": data["generated_utc"],
        "input": data["input"],
        "characteriser": data["characteriser"],
        "ai_diagnosis_notice": data["ai_diagnosis_notice"],
        "findings": findings,
    }


def write_html(result: AnalysisResult, data: dict, path: Path) -> None:
    s = data["summary"]
    ai = s["ai_categories"]
    used = classifier_used(result)
    informative_pct = 100 * s["frames_informative"] / max(s["frames_analysed"], 1)
    missed_tile = (
        f'<div class="tile alert"><b>{s["potentially_missed"]}</b>potentially missed</div>'
        if s["report_supplied"]
        else '<div class="tile"><b>&ndash;</b>no procedure report supplied</div>'
    )
    ai_tiles = ai_notice = classifier_meta = priority = ""
    if used:
        ai_tiles = (
            f'<div class="tile{" alert" if ai["cancerous"] else ""}"><b>{ai["cancerous"]}</b>suspected cancer (AI optical diagnosis)</div>'
            f'<div class="tile"><b>{ai[INDETERMINATE]}</b>'
            f'{"classifier not applied" if classifier_not_applied(result) else "indeterminate"} (no AI category)</div>'
        )
        ai_notice = f'<div class="notice">{esc(AI_DIAGNOSIS_NOTICE)}</div>'
        sha = (result.characteriser_info or {}).get("sha256")
        classifier_meta = (f" &middot; classifier {esc(result.characteriser)} v{esc(result.characteriser_version)}"
                           + (f" (SHA-256 {esc(_short_hash(sha))})" if sha else ""))
        ai_notice += _classifier_section(result)
        priority = _priority_list(result) if result.findings else ""
    cards = "".join(_finding_card(result, f) for f in result.findings) or "<p>No findings above threshold.</p>"
    blind_rows = "".join(
        f"<tr><td>{_fmt_time(b.start_s)}</td><td>{_fmt_time(b.end_s)}</td><td>{b.duration_s:.1f} s</td><td>{esc(b.reason)}</td></tr>"
        for b in result.blind_segments
    )
    unmatched_rows = "".join(
        f"<tr><td>{esc(r.report_id)}</td><td>{_fmt_time(r.time_s) if r.time_s is not None else 'frame ' + str(r.frame_index)}</td>"
        f"<td>{esc(r.location or '')}</td><td>{esc(r.note or '')}</td></tr>"
        for r in result.unmatched_reported
    )
    meta = _script_json(_report_meta(data))
    doc = f"""<!doctype html><html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1"><title>SecondLook Review</title><style>{CSS}</style></head>
<body><main>
<h1>Second-look review</h1>
<div class="muted">{esc(Path(result.input_path).name)} &middot; {result.config.modality.value} &middot;
detector {esc(result.detector)} v{esc(result.detector_version)}{classifier_meta} &middot; SecondLook-CRC v{__version__} &middot; {esc(data["generated_utc"][:19])} UTC</div>
<div class="notice">{esc(INTENDED_USE_NOTICE)}</div>
{ai_notice}
<div class="tiles">
  <div class="tile"><b>{_fmt_time(s["duration_s"])}</b>recording length</div>
  <div class="tile"><b>{s["findings"]}</b>findings flagged</div>
  {missed_tile}
  {ai_tiles}
  <div class="tile"><b>{informative_pct:.0f}%</b>frames of reviewable quality</div>
  <div class="tile"><b>{s["blind_segments"]}</b>blind segments ({s["blind_time_s"]:.0f} s)</div>
</div>
<h2>Timeline</h2>
{_timeline_svg(result)}
<div class="legend"><span><i style="background:var(--missed)"></i>potentially missed</span>
<span><i style="background:var(--accent)"></i>other finding</span><span><i style="background:var(--blind)"></i>blind segment</span>
<span><i style="background:var(--poor)"></i>poor-quality frame</span></div>
{priority}
<h2>Findings</h2>
{cards}
<h2>Blind segments</h2>
<p class="muted">Stretches where image quality was too poor to review. Lesions here can be missed by both the endoscopist and the AI.</p>
{f"<table><tr><th>From</th><th>To</th><th>Length</th><th>Reason</th></tr>{blind_rows}</table>" if blind_rows else "<p>None.</p>"}
{f"<h2>Reported lesions not found by the AI</h2><table><tr><th>ID</th><th>When</th><th>Location</th><th>Note</th></tr>{unmatched_rows}</table>" if unmatched_rows else ""}
<h2>Sign-off</h2>
<p><input type="text" id="reviewer" placeholder="Reviewer name / GMC number"> <button onclick="exportDecisions()">Export review decisions</button></p>
</main><script>const REPORT_META={meta};
{JS}</script></body></html>"""
    path.write_text(doc, encoding="utf-8")


def write(result: AnalysisResult, out_dir: Path | str) -> dict:
    out_dir = Path(out_dir)
    (out_dir / "findings").mkdir(parents=True, exist_ok=True)
    data = result_dict(result)
    (out_dir / "result.json").write_text(json.dumps(data, indent=2, allow_nan=False))
    for fid, imgs in result.images.items():
        for kind, img in imgs.items():
            cv2.imwrite(str(out_dir / "findings" / f"{fid}_{kind}.png"), img)
    write_html(result, data, out_dir / "report.html")
    return data
