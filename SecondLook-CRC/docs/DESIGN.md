# SecondLook-CRC — Software Design

> Working name. Offline AI second-look review for colonoscopy and capsule endoscopy recordings.
> Status: research prototype (v0.1). Not a medical device.

## 1. Purpose

Colonoscopy misses lesions. Adenoma miss rates in tandem studies are around one in four, and
most post-colonoscopy colorectal cancers (PCCRC) are thought to come from lesions that were
missed or not fully removed. Real-time CADe systems help during the procedure, but nobody
looks again *after* the procedure. Capsule endoscopy has the opposite problem: tens of
thousands of frames have to be read by a person after the fact, which is slow and tiring.

SecondLook is the step that is missing after the procedure. A clinician, trainee or researcher uploads a recorded
colonoscopy video or capsule study. The software then:

1. finds candidate lesions (polyps, possible neoplasia) the AI can see,
2. compares them with what the endoscopist reported and flags the **potentially missed** ones,
3. shows **where the recording was too poor to review** (blind segments),
4. explains each flag with a heatmap, the lesion outline, a filmstrip and plain-English reasons, and
5. records a second reader's decision for each flag. These decisions feed the report, the audit trail and model retraining.

It is designed as the shared "brain" between the company's products:

```
           VR-Caps simulator (this repo)          MADSyncro (simulator)
           synthetic labelled data  ─────┐        ┌──── training scenarios, trainee assessment
                                         ▼        ▼
                              ┌──────────────────────────┐
  Real recordings ──────────▶ │   SecondLook core models  │ ──▶ Second-look reports, audit, research
  (colonoscopy + capsule)     │  detect · explain · track │
                              └──────────────────────────┘
                                         │  same ONNX models, later real-time
                                         ▼
                              MADAlpha (robotic capsule) — future on-device detection
```

The detector is a versioned ONNX file, so the same model can later run in real time inside
MADSyncro and MADAlpha. Reviewer decisions on real cases become training labels. VR-Caps
and MADSyncro produce synthetic data for rare lesions and for capsule views that are hard
to collect. Together these close the loop.

## 2. Product characteristics → how the design meets them

| Characteristic (from the brief) | Design decision | Prototype status |
|---|---|---|
| No real-time / offline system | Batch pipeline over recorded files; no device connection | ✅ `pipeline.py` |
| AI trained on datasets | Pluggable detectors; training script exports ONNX + metadata sidecar | ✅ `training/`, `detectors/onnx_seg.py` |
| Post-evaluate a past procedure for missed findings | Findings are matched against the procedure report; unmatched ones are flagged *potentially missed* | ✅ `review.py` |
| Universal compatibility | Any video or image format OpenCV can read; image folders for capsule exports; vendor-neutral ONNX models | ✅ basic formats · ⏳ DICOM video, vendor capsule formats |
| Raw images/video → actionable insight | Quality gating → detection → tracking → one finding per lesion with timestamps | ✅ |
| Secondary analysis of real and synthetic data | Same pipeline runs on VR-Caps output; synthetic generator for tests | ✅ |
| Colonoscopy only, academic + clinical | Scope fixed to the lower GI tract; intended-use statement printed on every output | ✅ |
| Both traditional colonoscopy and capsule explicitly | Separate modality profiles (sampling, quality thresholds, tracking, report tolerance) | ✅ `config.py` |
| Visual explanation for every flag, not just a box | Evidence heatmap, exact outline, filmstrip and written rationale for every finding | ✅ `explain.py` |
| Fully local, on-premises at first (NHS Scotland) | No network calls; web UI binds to 127.0.0.1; audit log stored locally; inputs identified by hash | ✅ |
| Clinical **and** research/audit use | JSON output for research; HTML report for clinicians; tamper-evident audit log | ✅ |
| Second-look tool in the reporting pathway | Second-reader decision capture and sign-off export, linked to the procedure report | ✅ basic · ⏳ integration with endoscopy reporting systems |
| UK/EU regulatory clearance, then global | Built from the start with intended use, traceability, versioning and audit. See [REGULATORY.md](REGULATORY.md) | 📋 plan |

## 3. Architecture

```
 input (video / image folder / image)
   │
   ▼
 ingest.py ── subsample to analysis fps, resize, SHA-256 fingerprint
   │
   ▼
 quality.py ── blur / glare / darkness per frame ──▶ blind segments
   │ (informative frames only)
   ▼
 detectors/ ── HeuristicDetector (baseline, no training)
   │           OnnxSegmentationDetector (trained model)
   │           each detection = outline mask + evidence heatmap + named evidence
   ▼
 tracking.py ── link detections over time into findings (one per lesion)
   │
   ▼
 review.py ── compare with procedure report ──▶ reported / potentially missed / not compared
   │
   ▼
 characterise.py ── CADx plug-in (histology, malignancy risk) — empty until validated
   │
   ▼
 explain.py + report.py ── heatmap overlay, outline, filmstrip, rationale
   │                        report.html (self-contained) + result.json + PNGs
   ▼
 audit.py ── hash-chained JSONL: who, when, input hash, model version, summary
```

Interfaces: `cli.py` (scripting, batch audits, research) and `server.py` (local browser upload).

### Key design choices

* **Outline + heatmap, not boxes.** Segmentation-style detectors give the exact region and a
  per-pixel evidence map, so the explanation comes from the detector itself rather than from
  a separate saliency method added afterwards. Classifier-based models must provide Grad-CAM
  or a similar map to meet the `Detector` contract.
* **Findings, not frames.** Tracking turns hundreds of per-frame hits into one finding with a time
  range, the best frame and a filmstrip. That is what a reviewer can act on. Short tracks are
  dropped to cut single-frame noise.
* **Blind segments are a first-class output.** Neither a person nor an AI can see a lesion
  in blurred, flooded or dark footage. The report states how much of the recording could
  not be reviewed, which also supports quality-assurance audit.
* **Missed-finding logic is explicit and auditable.** A finding is *potentially missed* only
  when a procedure report was supplied and no reported lesion falls within the time tolerance
  of that finding. Matching is one to one. Reported lesions the AI did not find are listed too:
  they are AI false negatives and useful for validation.
* **Characterisation is deliberately off.** Malignancy-risk outputs carry the most clinical risk
  and need histology-labelled training data. The fields already exist in the outputs, but the
  report states "uncharacterised" until a validated model is plugged in.
* **ONNX as the model boundary.** It keeps runtime dependencies small, runs on CPU in hospital
  IT, and is portable to embedded targets (MADAlpha) and to the simulator (MADSyncro).

## 4. Modality differences

| | Traditional colonoscopy | Colon capsule endoscopy |
|---|---|---|
| Input | 25–60 fps video, 30–60 min | 4–35 fps adaptive, ~50k+ frames, hours |
| Control | Endoscopist steers, inflates, washes | Passive (or magnetically guided: MADAlpha) |
| Main miss cause | Lesion behind folds, fast withdrawal, poor prep | Frames not reviewed carefully, debris, lesion seen in 1–2 frames only |
| Profile | 5 fps analysis, tight tracking, 5 s report tolerance | every frame, loose tracking, 30 s tolerance |
| Extra value | Withdrawal-quality audit (blind time) | Shorter reading time: reviewer goes straight to flagged frames |

## 5. Roadmap

**Phase 0 — prototype (this code).** End-to-end pipeline, baseline detector, explanation,
missed-finding comparison, local UI, audit, synthetic tests.

**Phase 1 — trained detector (3–6 months).**
* Train on public data (Kvasir-SEG, CVC-ClinicDB, PolypGen, SUN, REAL-Colon) with VR-Caps pre-training. See [DATASETS.md](DATASETS.md).
* Swap the compact U-Net for a pretrained-encoder model (e.g. Polyp-PVT / PraNet-style, or YOLO-seg).
* Evaluate per lesion, not per frame: sensitivity per polyp, false positives per minute, time-to-first-detection; split by patient and by centre.
* Capsule model trained separately (Kvasir-Capsule, plus colon-capsule data from partners).

**Phase 2 — clinical pilot, on-premises (NHS Scotland).**
* DICOM / vendor export ingestion. Removal of on-screen patient identifiers (burned-in overlay masking).
* Integration with the endoscopy reporting system (e.g. HL7/FHIR import of the procedure report, so it does not have to be written by hand as JSON).
* Login, roles (trainee / consultant / researcher), multi-user audit, job queue for long videos.
* Retrospective study: run on archived procedures that had a PCCRC or a surveillance finding and measure what SecondLook would have flagged.

**Phase 3 — characterisation and regulated release.**
* CADx head (adenoma vs. hyperplastic; NICE/JNET-style features; Paris morphology), trained on histology-confirmed data.
* Size estimation (depth from monocular images — VR-Caps already provides depth ground truth).
* Location / coverage mapping (which segments were seen), building on VR-Caps pose and depth work.
* UKCA / CE marking ([REGULATORY.md](REGULATORY.md)).

**Phase 4 — real-time in MADSyncro / MADAlpha.** Optimised ONNX/TensorRT models running live.
SecondLook remains the offline QA and training-data hub.

## 6. Commercial and partnership use

* **Trainees and educators:** look back at your own procedures, with an explanation for every flag. MADSyncro scores trainees with the same model.
* **Consultants and endoscopy units:** second reader in the reporting pathway; blind-time and missed-lesion audit (supports JAG quality assurance).
* **Researchers and universities:** batch JSON output, reproducible versioned models and audit trail; a route for partners to contribute data and co-develop models under data-sharing agreements.
* **Licensing:** on-premises licence per site at first, then a hosted option once security and regulatory approval allow.

## 7. Known limitations of the prototype

* The baseline heuristic detector has only been checked on synthetic data. It **will** produce false positives and misses on real footage. Use a trained model.
* No mm size estimate, no anatomical location, no histology prediction yet.
* DICOM and proprietary capsule formats are not read yet. Export to MP4/AVI or images first.
* The web UI has no authentication. It is for single-user local use only.
