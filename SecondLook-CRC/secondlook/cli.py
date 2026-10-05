"""Command line entry point.

    secondlook analyse RECORDING [--modality colonoscopy|capsule] [--report report.json] [--out DIR]
    secondlook demo [--out DIR]
    secondlook serve [--port 8765]
    secondlook audit-verify [--log PATH]
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from . import INTENDED_USE_NOTICE, __version__, audit, report, synthetic
from .config import get_config
from .detectors import load_detector
from .ingest import input_fingerprint
from .models import Modality
from .pipeline import analyse
from .review import load_report

DEFAULT_AUDIT_LOG = Path.home() / ".secondlook" / "audit.jsonl"


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
) -> dict:
    overrides = {"detection_threshold": threshold} if threshold is not None else {}
    cfg = get_config(modality, **overrides)
    detector = load_detector(model_path)
    reported = load_report(report_path) if report_path else None
    result = analyse(input_path, cfg, detector, reported=reported, image_sequence_fps=image_fps)
    data = report.write(result, out_dir)
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
    a.add_argument("--image-fps", type=float, default=1.0, help="capture rate of image-folder inputs (capsule frames/s)")
    a.add_argument("--threshold", type=float, help="override detection threshold (0-1)")
    a.add_argument("--out", type=Path, help="output directory (default: <input>_secondlook)")
    a.add_argument("--audit-log", type=Path, default=DEFAULT_AUDIT_LOG)
    a.add_argument("--user", help="name recorded in the audit log (default: OS user)")

    d = sub.add_parser("demo", help="generate a synthetic recording and analyse it")
    d.add_argument("--out", type=Path, default=Path("demo_output"))
    d.add_argument("--modality", choices=[m.value for m in Modality], default=Modality.COLONOSCOPY.value)

    s = sub.add_parser("serve", help="local web interface for uploading recordings (binds to localhost only)")
    s.add_argument("--port", type=int, default=8765)
    s.add_argument("--data-dir", type=Path, default=Path.home() / ".secondlook" / "runs")

    v = sub.add_parser("audit-verify", help="check the audit log has not been altered")
    v.add_argument("--log", type=Path, default=DEFAULT_AUDIT_LOG)

    args = parser.parse_args(argv)
    print(INTENDED_USE_NOTICE, file=sys.stderr)

    if args.cmd == "analyse":
        out = args.out or args.input.with_name(args.input.stem + "_secondlook")
        data = run_analysis(
            args.input, out, args.modality, args.report, args.model, args.image_fps, args.audit_log, args.user, args.threshold
        )
        _print_summary(data, out)
    elif args.cmd == "demo":
        args.out.mkdir(parents=True, exist_ok=True)
        capsule = args.modality == Modality.CAPSULE.value
        fps = 2.0 if capsule else 10.0
        rec = args.out / ("synthetic_capsule_frames" if capsule else "synthetic_colonoscopy.avi")
        synthetic.generate(rec, duration_s=200 if capsule else 40, fps=fps, as_frames=capsule,
                           blur_interval_s=(100, 130) if capsule else (20, 27))
        data = run_analysis(rec, args.out / "review", args.modality, args.out / f"{rec.stem}_report.json",
                            image_fps=fps, audit_log=args.out / "audit.jsonl")
        _print_summary(data, args.out / "review")
    elif args.cmd == "serve":
        from .server import serve

        serve(args.port, args.data_dir)
    elif args.cmd == "audit-verify":
        ok, msg = audit.verify(args.log)
        print(msg)
        return 0 if ok else 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
