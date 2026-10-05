"""Command line entry point.

    secondlook analyse RECORDING [--modality colonoscopy|capsule] [--report report.json] [--out DIR]
                                 [--model SEG.onnx] [--classifier CADX.onnx]
    secondlook demo [--out DIR] [--classifier CADX.onnx]
    secondlook serve [--port 8765] [--classifier CADX.onnx]
    secondlook audit-verify [--log PATH]
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from . import INTENDED_USE_NOTICE, __version__, audit, report, synthetic
from .characterise import Characteriser, NullCharacteriser
from .config import get_config
from .detectors import load_detector
from .diagnosis import CATEGORIES, DISPLAY_NAME, load_classifier
from .diagnosis.classifier import SAFETY_STATEMENT
from .ingest import input_fingerprint
from .models import Modality
from .pipeline import analyse
from .review import load_report

DEFAULT_AUDIT_LOG = Path.home() / ".secondlook" / "audit.jsonl"
# Seed of the colonoscopy demo recording drawn when a classifier is given: the baseline
# detector finds all three lesions in it.
DEMO_LESION_SEED = 1
CLASSIFIER_HELP = (
    "trained ONNX lesion classifier (benign / precancerous / cancerous) with its .json sidecar; "
    "without it findings are detected but not characterised"
)


def run_analysis(
    input_path: Path,
    out_dir: Path,
    modality: str,
    report_path: Path | None = None,
    model_path: str | None = None,
    image_fps: float = 1.0,
    audit_log: Path = DEFAULT_AUDIT_LOG,
    user: str | None = None,
    threshold: float | None = None,
    classifier_path: Path | str | None = None,
    characteriser: Characteriser | None = None,
) -> dict:
    """`characteriser` lets a long-running caller (the web server) reuse an
    already-loaded classifier; otherwise one is loaded from `classifier_path`."""
    overrides = {"detection_threshold": threshold} if threshold is not None else {}
    cfg = get_config(modality, **overrides)
    detector = load_detector(model_path)
    if characteriser is None:
        characteriser = load_classifier(classifier_path) if classifier_path else NullCharacteriser()
    reported = load_report(report_path) if report_path else None
    result = analyse(
        input_path, cfg, detector, reported=reported, characteriser=characteriser, image_sequence_fps=image_fps
    )
    data = report.write(result, out_dir)
    info = result.characteriser_info
    audit.append(
        audit_log,
        "analysis",
        user=user,
        software_version=__version__,
        input_name=input_path.name,
        input_sha256=input_fingerprint(input_path),
        procedure_report_supplied=reported is not None,
        modality=cfg.modality.value,
        detector=detector.name,
        detector_version=detector.version,
        characteriser=characteriser.name,
        characteriser_version=characteriser.version,
        characteriser_sha256=info.get("sha256"),
        characteriser_sidecar_sha256=info.get("sidecar_sha256"),
        characteriser_not_applied=info.get("unsupported_reason"),
        output_dir=str(out_dir.resolve()),
        summary=data["summary"],
    )
    return data


def _print_summary(data: dict, out_dir: Path) -> None:
    s = data["summary"]
    print(f"Analysed {s['frames_analysed']} frames ({s['frames_informative']} reviewable) in {s['runtime_s']:.1f} s")
    print(f"Findings: {s['findings']}", end="")
    print(f"  (potentially missed: {s['potentially_missed']})" if s["report_supplied"] else "  (no procedure report supplied)")
    print(f"Blind segments: {s['blind_segments']} ({s['blind_time_s']:.0f} s)")
    chars = [c for c in (f.get("characterisation") or {} for f in data["findings"]) if c.get("model")]
    clf = data.get("characteriser") or {}
    if clf.get("unsupported_reason"):
        print(f"WARNING: lesion classifier not applied: {clf['unsupported_reason']}.")
    if chars:
        sha = f", SHA-256 {clf['sha256'][:12]}" if clf.get("sha256") else ""
        print(f"AI optical diagnosis per finding ({chars[0]['model']} {chars[0].get('model_version')}{sha}):")
        counts = data["summary"]["ai_categories"]
        for cat in CATEGORIES:
            print(f"  {DISPLAY_NAME[cat]}: {counts[cat]}")
        print(f"  No AI category (uncertain): {counts[report.INDETERMINATE]}")
        if clf.get("calibrated") is not True:
            print("  Probabilities are not calibrated: read them as a ranking, not as risks.")
        if clf.get("validation_status"):
            print(f"  {clf['validation_status']}")
        print(f"  {SAFETY_STATEMENT}")
    print(f"Report: {out_dir / 'report.html'}")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="secondlook", description="Offline second-look review of colonoscopy and capsule recordings.")
    parser.add_argument("--version", action="version", version=__version__)
    sub = parser.add_subparsers(dest="cmd", required=True)

    a = sub.add_parser("analyse", help="analyse a video, image folder or image")
    a.add_argument("input", type=Path)
    a.add_argument("--modality", choices=[m.value for m in Modality], default=Modality.COLONOSCOPY.value)
    a.add_argument("--report", type=Path, help="procedure report JSON listing the lesions the endoscopist found")
    a.add_argument("--model", help="trained ONNX segmentation model (default: training-free baseline)")
    a.add_argument("--classifier", help=CLASSIFIER_HELP)
    a.add_argument("--image-fps", type=float, default=1.0, help="capture rate of image-folder inputs (capsule frames/s)")
    a.add_argument("--threshold", type=float, help="override detection threshold (0-1)")
    a.add_argument("--out", type=Path, help="output directory (default: <input>_secondlook)")
    a.add_argument("--audit-log", type=Path, default=DEFAULT_AUDIT_LOG)
    a.add_argument("--user", help="name recorded in the audit log (default: OS user)")

    d = sub.add_parser("demo", help="generate a synthetic recording and analyse it")
    d.add_argument("--out", type=Path, default=Path("demo_output"))
    d.add_argument("--modality", choices=[m.value for m in Modality], default=Modality.COLONOSCOPY.value)
    d.add_argument("--classifier", help=CLASSIFIER_HELP + "; the colonoscopy demo recording then has one "
                   "benign, one precancerous and one cancerous synthetic lesion")

    s = sub.add_parser("serve", help="local web interface for uploading recordings (binds to localhost only)")
    s.add_argument("--port", type=int, default=8765)
    s.add_argument("--data-dir", type=Path, default=Path.home() / ".secondlook" / "runs")
    s.add_argument("--classifier", help=CLASSIFIER_HELP)

    v = sub.add_parser("audit-verify", help="check the audit log has not been altered")
    v.add_argument("--log", type=Path, default=DEFAULT_AUDIT_LOG)

    args = parser.parse_args(argv)
    print(INTENDED_USE_NOTICE, file=sys.stderr)

    if args.cmd == "analyse":
        out = args.out or args.input.with_name(args.input.stem + "_secondlook")
        data = run_analysis(
            args.input, out, args.modality, args.report, args.model, args.image_fps, args.audit_log, args.user,
            args.threshold, classifier_path=args.classifier,
        )
        _print_summary(data, out)
    elif args.cmd == "demo":
        characteriser = load_classifier(args.classifier) if args.classifier else None  # fail before rendering
        args.out.mkdir(parents=True, exist_ok=True)
        capsule = args.modality == Modality.CAPSULE.value
        fps = 2.0 if capsule else 10.0
        rec = args.out / ("synthetic_capsule_frames" if capsule else "synthetic_colonoscopy.avi")
        if characteriser and not capsule:
            # One lesion of each category, with room for all three outside the blurred stretch.
            scene = dict(duration_s=60, blur_interval_s=(30, 37), lesion_kinds=list(CATEGORIES), seed=DEMO_LESION_SEED)
        else:
            scene = dict(duration_s=200 if capsule else 40, blur_interval_s=(100, 130) if capsule else (20, 27))
        synthetic.generate(rec, fps=fps, as_frames=capsule, **scene)
        data = run_analysis(rec, args.out / "review", args.modality, args.out / f"{rec.stem}_report.json",
                            image_fps=fps, audit_log=args.out / "audit.jsonl", characteriser=characteriser)
        _print_summary(data, args.out / "review")
    elif args.cmd == "serve":
        from .server import serve

        serve(args.port, args.data_dir, args.classifier)
    elif args.cmd == "audit-verify":
        ok, msg = audit.verify(args.log)
        print(msg)
        return 0 if ok else 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
