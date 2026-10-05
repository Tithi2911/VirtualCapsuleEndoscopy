import json
import threading
import urllib.request
from http.server import ThreadingHTTPServer

import pytest

from secondlook import audit, synthetic
from secondlook.cli import run_analysis
from secondlook.config import get_config
from secondlook.models import FindingStatus, ReportedFinding
from secondlook.review import compare
from secondlook.server import make_handler
from secondlook.tracking import iou


@pytest.fixture(scope="module")
def colonoscopy_run(tmp_path_factory):
    d = tmp_path_factory.mktemp("colo")
    video = d / "synthetic_colonoscopy.avi"
    truth = synthetic.generate(video, duration_s=40, fps=10)
    data = run_analysis(video, d / "out", "colonoscopy", d / "synthetic_colonoscopy_report.json",
                        audit_log=d / "audit.jsonl")
    return d, truth, data


def _overlaps(finding, polyp, slack=1.0):
    return finding["first_s"] <= polyp["visible_to_s"] + slack and finding["last_s"] >= polyp["visible_from_s"] - slack


def test_each_synthetic_polyp_is_found(colonoscopy_run):
    _, truth, data = colonoscopy_run
    for polyp in truth["polyps"]:
        assert any(_overlaps(f, polyp) for f in data["findings"]), polyp


def test_unreported_polyp_is_flagged_as_potentially_missed(colonoscopy_run):
    _, truth, data = colonoscopy_run
    p1, p2 = truth["polyps"]
    assert any(_overlaps(f, p1) and f["status"] == "reported" for f in data["findings"])
    assert any(_overlaps(f, p2) and f["status"] == "potentially_missed" for f in data["findings"])


def test_blurred_stretch_is_reported_as_blind_segment(colonoscopy_run):
    _, truth, data = colonoscopy_run
    start, end = truth["blur_interval_s"]
    assert any(b["start_s"] <= start + 0.5 and b["end_s"] >= end - 0.5 for b in data["blind_segments"])


def test_every_finding_has_visual_and_written_explanation(colonoscopy_run):
    d, _, data = colonoscopy_run
    for f in data["findings"]:
        assert len(f["rationale"]) >= 3
        for kind in ("overlay", "original", "filmstrip"):
            assert (d / "out" / "findings" / f"{f['id']}_{kind}.png").exists()
    html = (d / "out" / "report.html").read_text()
    assert "Why this was flagged" in html and "not a medical device" in html


def test_audit_log_is_written_and_tamper_evident(colonoscopy_run):
    d, _, _ = colonoscopy_run
    log = d / "audit.jsonl"
    audit.append(log, "test-event", user="tester")
    assert audit.verify(log)[0]
    lines = log.read_text().splitlines()
    entry = json.loads(lines[0])
    entry["summary"]["findings"] = 0
    lines[0] = json.dumps(entry, sort_keys=True)
    log.write_text("\n".join(lines) + "\n")
    ok, msg = audit.verify(log)
    assert not ok and "line 1" in msg


def test_capsule_frame_folder(tmp_path):
    frames = tmp_path / "synthetic_capsule_frames"
    truth = synthetic.generate(frames, duration_s=200, fps=2.0, as_frames=True, blur_interval_s=None)
    data = run_analysis(frames, tmp_path / "out", "capsule", image_fps=2.0, audit_log=tmp_path / "a.jsonl")
    assert data["modality"] == "capsule"
    assert all(f["status"] == "unreviewed" for f in data["findings"])  # no procedure report supplied
    for polyp in truth["polyps"]:
        assert any(_overlaps(f, polyp, slack=2.0) for f in data["findings"]), polyp


def test_report_matching_is_one_to_one():
    from secondlook.models import Detection, Finding

    def finding(fid, t0, t1):
        dets = [Detection(int(t * 10), t, (0, 0, 10, 10), 0.9, "x", None, None) for t in (t0, t1)]
        return Finding(fid, dets)

    cfg = get_config("colonoscopy")
    a, b = finding("F1", 10, 12), finding("F2", 13, 14)
    unmatched = compare([a, b], [ReportedFinding("R1", time_s=11)], cfg)
    assert a.status == FindingStatus.REPORTED and a.matched_report_id == "R1"
    assert b.status == FindingStatus.POTENTIALLY_MISSED
    assert unmatched == []


def test_iou():
    assert iou((0, 0, 10, 10), (0, 0, 10, 10)) == 1.0
    assert iou((0, 0, 10, 10), (20, 20, 5, 5)) == 0.0


def test_local_server_upload(tmp_path):
    img = tmp_path / "frames"
    synthetic.generate(img, duration_s=4, fps=2.0, as_frames=True, blur_interval_s=None)
    server = ThreadingHTTPServer(("127.0.0.1", 0), make_handler(tmp_path / "runs"))
    threading.Thread(target=server.serve_forever, daemon=True).start()
    try:
        base = f"http://127.0.0.1:{server.server_address[1]}"
        assert b"upload a recording" in urllib.request.urlopen(base + "/").read()
        body = (img / "frame_00000.png").read_bytes()
        req = urllib.request.Request(base + "/analyse", data=body, method="POST",
                                     headers={"X-Filename": "frame.png", "X-Modality": "capsule"})
        result = json.loads(urllib.request.urlopen(req).read())
        assert b"Second-look review" in urllib.request.urlopen(base + result["report"]).read()
    finally:
        server.shutdown()
