# SecondLook-CRC

**Offline AI second-look review for colonoscopy and capsule endoscopy recordings.**

> ⚠️ Research prototype. Not a medical device. Do not use for diagnosis or clinical decisions.

Upload a recorded colonoscopy video or capsule study. SecondLook then:

* **finds candidate lesions** and groups them over time, so you get one finding per lesion, not hundreds of boxes,
* **compares them with the procedure report** and flags the ones the endoscopist did not document as *potentially missed*,
* **shows blind segments**, the parts of the recording too blurred, dark or glary to review,
* **explains every flag** with an evidence heatmap, the exact lesion outline, a filmstrip of frames and plain-English reasons,
* **captures the second reader's decision** (confirmed / not a lesion / uncertain) and exports it for sign-off, audit and retraining,
* runs **fully on the local machine**, with a tamper-evident audit log.

It handles **traditional colonoscopy** and **capsule endoscopy** through separate modality
profiles. It is designed to become the shared detection "brain" for MADSyncro (simulator)
and MADAlpha (robotic capsule), with VR-Caps (this repository) supplying synthetic training data.

![Example report](docs/img/report_example.jpg)

## Quick start

```sh
cd SecondLook-CRC
pip install -e '.[dev]'

# Generate a synthetic recording with two polyps (the "report" only mentions one) and review it
secondlook demo --out demo_output
#   -> demo_output/review/report.html : one finding matches the report, one is flagged "potentially missed"

# Analyse your own recording
secondlook analyse procedure.mp4 --modality colonoscopy --report procedure_report.json
secondlook analyse capsule_frames/ --modality capsule --image-fps 4

# Local browser interface (localhost only)
secondlook serve        # open http://127.0.0.1:8765

# Check the audit log has not been altered
secondlook audit-verify
```

The procedure report is a simple JSON list of the lesions the endoscopist found, by time or
frame. See [examples/procedure_report.json](examples/procedure_report.json). Without it, findings are
listed as "not compared".

## Using a trained model

The default detector is a training-free colour/texture baseline. It exists so the whole
workflow can be shown and tested, and it is **not** accurate on real footage. Train a
segmentation model and pass it in:

```sh
pip install -e '.[train,model]'
python training/train_segmentation.py --data kvasir_seg --epochs 40 \
    --out models/unet.pt --export models/unet.onnx
secondlook analyse procedure.mp4 --model models/unet.onnx
```

Dataset folders use `images/` + `masks/` (Kvasir-SEG layout). VR-Caps synthetic exports can be
used for pre-training. See [docs/DATASETS.md](docs/DATASETS.md).

## Outputs

| File | Use |
|---|---|
| `report.html` | Self-contained clinician report (images embedded, no server needed) with second-reader controls |
| `result.json` | Machine-readable results for research and audit |
| `findings/F001_overlay.png` … | Heatmap+outline, original frame and filmstrip for each finding |
| `~/.secondlook/audit.jsonl` | Hash-chained log: user, time, input SHA-256, model version, summary |

## Documentation

* [docs/DESIGN.md](docs/DESIGN.md): purpose, how each product characteristic is met, architecture, roadmap, commercial use
* [docs/DATASETS.md](docs/DATASETS.md): open and restricted datasets, licensing, VR-Caps synthetic data, evaluation rules
* [docs/REGULATORY.md](docs/REGULATORY.md): UKCA / CE (MDR) / AI Act plan, standards, NHS Scotland deployment

## Project layout

```
secondlook/
  ingest.py        video / image-folder loading, fingerprinting
  quality.py       per-frame quality, blind segments
  detectors/       baseline heuristic + ONNX segmentation model
  tracking.py      detections -> findings over time
  review.py        compare with the procedure report (missed-finding logic)
  characterise.py  CADx plug-in point (disabled until validated)
  explain.py       heatmap, outline, filmstrip, written rationale
  report.py        HTML + JSON report
  audit.py         tamper-evident audit log
  server.py        local upload UI
  synthetic.py     demo recordings with ground truth
training/          model training and ONNX export
tests/             end-to-end tests on synthetic data
```

Run the tests with `pytest`.
