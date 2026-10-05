"""Report display of the AI optical diagnosis, built from a hand-made AnalysisResult (no model needed)."""

import json
import re

import numpy as np
import pytest

from secondlook import explain, report
from secondlook.config import get_config
from secondlook.models import Characterisation, Detection, Finding, FindingStatus
from secondlook.pipeline import AnalysisResult

EVIL_NAME = "<script>x</script>"
EVIL_VERSION = '"><img src=x onerror=alert(1)>'


def _char(category, probs, abstain_reason=None, frames=6, agreement=1.0, explained=False, calibrated=None):
    rng = np.random.default_rng(1)
    c = Characterisation(
        model=EVIL_NAME,
        model_version=EVIL_VERSION,
        category=category,
        probabilities=probs,
        confidence=max(probs.values()) if probs else None,
        abstained=category is None,
        abstain_reason=abstain_reason,
        frames_used=frames if probs else 0,
        frame_agreement=agreement if probs else None,
        malignancy_risk=probs["cancerous"] if probs else None,
        neoplasia_risk=probs["precancerous"] + probs["cancerous"] if probs else None,
        calibrated=calibrated,
        note="AI optical diagnosis (test).",
    )
    if explained:
        c.explanation_crop = rng.integers(0, 255, (64, 64, 3), dtype=np.uint8)
        heat = np.zeros((64, 64), np.float32)
        heat[20:40, 20:40] = 1.0
        c.explanation_heatmap = heat
    return c


def _finding(fid, t, char, status=FindingStatus.REPORTED):
    dets = [
        Detection(int(t * 10) + i, t + 0.1 * i, (40, 30, 50, 40), 0.8 + 0.01 * i, "lesion", None, None, {"redness_sd": 1.2})
        for i in range(3)
    ]
    return Finding(fid, dets, status=status, characterisation=char, rationale=["one", "two", "three"])


def _images(c: Characterisation):
    img = np.random.default_rng(0).integers(0, 255, (120, 160, 3), dtype=np.uint8)
    out = {"original": img, "overlay": img.copy(), "filmstrip": explain.filmstrip([img, img])}
    if c.explanation_crop is not None:
        out["diagnosis"] = explain.diagnosis_overlay(c.explanation_crop, c.explanation_heatmap)
    return out


def _result(findings, characteriser=EVIL_NAME, version=EVIL_VERSION, info=None, modality="colonoscopy"):
    return AnalysisResult(
        input_path="/data/procedure.avi",
        config=get_config(modality),
        detector="heuristic",
        detector_version="1",
        characteriser=characteriser,
        characteriser_version=version,
        frames_analysed=100,
        frames_informative=90,
        duration_s=60.0,
        findings=findings,
        blind_segments=[],
        unmatched_reported=[],
        report_supplied=True,
        timeline=[(i * 0.6, True) for i in range(100)],
        runtime_s=1.0,
        images={f.finding_id: _images(f.characterisation) for f in findings},
        characteriser_info=info or {},
    )


@pytest.fixture()
def classified(tmp_path):
    findings = [
        _finding("F001", 5, _char("benign", {"benign": 0.926, "precancerous": 0.07, "cancerous": 0.004}, explained=True)),
        _finding(
            "F002", 15,
            _char("cancerous", {"benign": 0.07, "precancerous": 0.21, "cancerous": 0.72}, frames=8, agreement=0.75,
                  explained=True, calibrated=True),
            status=FindingStatus.POTENTIALLY_MISSED,
        ),
        _finding("F003", 25, _char(None, {"benign": 0.30, "precancerous": 0.45, "cancerous": 0.25},
                                   abstain_reason="low confidence (0.45 < 0.60)", agreement=0.5)),
        _finding("F004", 35, Characterisation()),
        _finding("F005", 45, _char(None, {}, abstain_reason="classifier error (RuntimeError: boom)")),
    ]
    data = report.write(_result(findings), tmp_path)
    return tmp_path, data, (tmp_path / "report.html").read_text()


def _cards(html: str) -> dict[str, str]:
    return {re.search(r'id="(F\d+)"', c).group(1): c for c in html.split('<section class="card"')[1:]}


def test_json_summary_counts_ai_categories(classified):
    out, data, _ = classified
    assert data["summary"]["ai_categories"] == {
        "benign": 1, "precancerous": 0, "cancerous": 1, "indeterminate": 2, "uncharacterised": 1,
    }
    assert [f["ai_category"] for f in data["findings"]] == [
        "benign", "cancerous", "indeterminate", "uncharacterised", "indeterminate",
    ]
    assert data["characteriser"] == {"name": EVIL_NAME, "version": EVIL_VERSION}
    for key in ("findings", "potentially_missed", "blind_segments", "report_supplied", "runtime_s"):
        assert key in data["summary"]
    saved = json.loads((out / "result.json").read_text())
    assert saved["summary"]["ai_categories"] == data["summary"]["ai_categories"]
    char = saved["findings"][1]["characterisation"]
    assert char["probabilities"]["cancerous"] == pytest.approx(0.72)
    assert "explanation_crop" not in char and "explanation_heatmap" not in char
    assert (out / "findings" / "F002_diagnosis.png").exists()


def test_html_shows_badges_probabilities_and_explanation(classified):
    _, _, html = classified
    cards = _cards(html)
    f2 = cards["F002"]
    assert "AI optical diagnosis" in f2 and 'class="cat cancerous">Suspected cancer<' in f2
    for pct in ("7%", "21%", "72%"):
        assert f'<td class="pct">{pct}</td>' in f2
    assert 'style="width:72.0%"' in f2 and 'aria-label="AI category probabilities for F002"' in f2
    assert "75% (6 of 8 frames)" in f2
    assert "occlusion sensitivity" in f2 and "not a pathology image" in f2 and "Image regions that most" in f2
    assert "<caption>Calibrated probability per category, averaged over 8 frames</caption>" in f2

    f1 = cards["F001"]
    assert 'class="cat benign">Benign (non-neoplastic)<' in f1
    assert "Calibrated probability" not in f1 and "(not calibrated: read as a ranking, not as a risk)" in f1
    assert '<td class="pct">&lt;1%</td>' in f1 and '<td class="pct">93%</td>' in f1

    f3 = cards["F003"]
    assert "Indeterminate (AI abstained)" in f3 and "low confidence (0.45 &lt; 0.60)" in f3
    assert '<td class="pct">45%</td>' in f3 and "occlusion sensitivity" not in f3

    f5 = cards["F005"]
    assert "Indeterminate (AI abstained)" in f5 and "classifier error (RuntimeError: boom)" in f5

    f4 = cards["F004"]
    assert "AI optical diagnosis" not in f4 and "uncharacterised" in f4


def test_safety_line_next_to_every_ai_diagnosis(classified):
    _, data, html = classified
    cards = _cards(html)
    for fid in ("F001", "F002", "F003", "F005"):
        assert report.AI_DIAGNOSIS_NOTICE in cards[fid]
    assert "resect-and-discard" in report.AI_DIAGNOSIS_NOTICE and "histopathology" in report.AI_DIAGNOSIS_NOTICE
    assert "including ones that are not lesions" in report.AI_DIAGNOSIS_NOTICE
    assert data["ai_diagnosis_notice"] == report.AI_DIAGNOSIS_NOTICE
    assert "not a medical device" in html


def test_priority_list_puts_findings_without_a_category_above_benign(classified):
    _, _, html = classified
    section = html.split("<h2>Priority list</h2>", 1)[1].split("<h2>Findings</h2>", 1)[0]
    order = re.findall(r'<li><a href="#(F\d+)">', section)
    # Suspected cancer, then no AI category (with an estimate first), then benign.
    assert order == ["F002", "F003", "F004", "F005", "F001"]
    assert "AI P(suspected cancer) 72%" in section and "P(neoplastic) 93%" in section
    assert "no AI estimate" in section and "needs human review" in section and "potentially missed" in section
    assert '<div class="tile alert"><b>1</b>suspected cancer' in html


def test_untrusted_model_text_is_escaped(classified):
    _, _, html = classified
    assert EVIL_NAME not in html and "<img src=x" not in html
    assert "&lt;script&gt;x&lt;/script&gt;" in html
    assert "&quot;&gt;&lt;img src=x onerror=alert(1)&gt;" in html
    meta = json.loads(re.search(r"const REPORT_META=(.*?);\n", html).group(1))
    assert meta["characteriser"]["name"] == EVIL_NAME  # intact once decoded, inert inside the script element


def test_export_includes_reviewer_category_histology_and_ai_output(classified):
    _, _, html = classified
    f2 = _cards(html)["F002"]
    assert 'name="rc-F002" class="reviewer-category"' in f2 and '<option value="cancerous">Suspected cancer</option>' in f2
    assert '<option value="unsure">Unsure</option>' in f2
    assert "Histology result (when available)" in f2 and 'name="h-F002" class="histology"' in f2
    js = html.split("function exportDecisions", 1)[1]
    for key in ("reviewer_category:", "histology:", "ai_category:", "ai_probabilities:"):
        assert key in js
    meta = json.loads(re.search(r"const REPORT_META=(.*?);\n", html).group(1))
    assert meta["findings"]["F002"]["ai_category"] == "cancerous"
    assert meta["findings"]["F002"]["probabilities"]["cancerous"] == pytest.approx(0.72)
    assert meta["findings"]["F003"]["ai_category"] == "indeterminate"
    assert meta["findings"]["F004"]["ai_category"] == "uncharacterised"


def test_non_finite_ai_output_is_indeterminate_and_json_stays_strict(tmp_path):
    nan = float("nan")
    broken = _char("benign", {"benign": nan, "precancerous": nan, "cancerous": nan})
    data = report.write(_result([_finding("F001", 5, broken)], modality="capsule"), tmp_path)
    assert data["findings"][0]["ai_category"] == "indeterminate"
    text = (tmp_path / "result.json").read_text()
    saved = json.loads(text, parse_constant=lambda c: pytest.fail(f"non-standard JSON constant {c}"))
    assert saved["findings"][0]["characterisation"]["probabilities"]["benign"] is None
    assert saved["config"]["analysis_fps"] is None  # capsule: every frame (infinite rate)
    card = _cards((tmp_path / "report.html").read_text())["F001"]
    assert "Indeterminate (AI abstained)" in card and "not numbers" in card
    assert "nan%" not in card and "width:nan" not in card and '<td class="pct">n/a</td>' in card


def test_classifier_provenance_and_validation_status_are_shown(tmp_path):
    info = {
        "sha256": "a" * 64, "sidecar_sha256": "b" * 64, "calibrated": False, "modalities": ["colonoscopy"],
        "training_data": ["synth_lesions"], "synthetic_training_data": True,
        "validation_status": "Trained on synthetic images only: a software test, not a clinical model.",
        "validation_warnings": ["The validation set has fewer than 30 examples of cancerous"],
        "intended_use": "Research use only. Not a medical device.",
    }
    findings = [_finding("F001", 5, _char("benign", {"benign": 0.9, "precancerous": 0.07, "cancerous": 0.03}))]
    data = report.write(_result(findings, info=info), tmp_path)
    html = (tmp_path / "report.html").read_text()
    assert data["characteriser"]["sha256"] == "a" * 64 and data["characteriser"]["name"] == EVIL_NAME
    section = html.split('<section class="classifier"', 1)[1].split("</section>", 1)[0]
    for text in ("a" * 64, "b" * 64, "not calibrated", "synth_lesions", "synthetic images only",
                 "fewer than 30 examples", "Research use only. Not a medical device."):
        assert text in section
    card = _cards(html)["F001"]
    assert "SHA-256 aaaaaaaaaaaa" in card and "synthetic images only" in card
    assert "(SHA-256 aaaaaaaaaaaa)" in html.split("<h1>", 1)[1].split("</div>", 1)[0]


def test_without_classifier_report_stays_uncharacterised(tmp_path):
    findings = [_finding("F001", 5, Characterisation()), _finding("F002", 15, Characterisation())]
    data = report.write(_result(findings, characteriser="none", version="-"), tmp_path)
    html = (tmp_path / "report.html").read_text()
    assert data["summary"]["ai_categories"]["uncharacterised"] == 2
    assert sum(data["summary"]["ai_categories"].values()) == 2
    assert "AI optical diagnosis" not in html and "Priority list" not in html
    assert "suspected cancer (AI" not in html and report.AI_DIAGNOSIS_NOTICE not in html
    assert data["ai_diagnosis_notice"] is None
    assert "No lesion classifier configured" in html
    assert "Reviewer optical diagnosis" in html  # reviewer can still record their own optical diagnosis
