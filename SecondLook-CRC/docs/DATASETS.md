# Datasets

**Always check the current licence and terms before using a dataset.** Many public
endoscopy datasets allow **research / non-commercial use only**. A model trained on them may
not be usable in a commercial product without permission from the dataset owners. Keep a
record of which datasets each released model was trained on. The ONNX sidecar's
`training_data` field and the audit log support this.

The **Histology** column marks datasets with a pathology result per polyp. Only those can
train or test the lesion classifier (benign / precancerous / cancerous; see
[DIAGNOSIS.md](DIAGNOSIS.md)). The others help detection and segmentation only.

## Colonoscopy — open or by registration

| Dataset | Content | Histology | Notes |
|---|---|---|---|
| Kvasir-SEG | 1,000 polyp images with masks | No | Standard segmentation benchmark |
| HyperKvasir | ~110k images (10k labelled), 374 videos, GI findings | No | Upper and lower GI. Includes polyps and quality classes (bowel prep) |
| CVC-ClinicDB | 612 frames with masks, from colonoscopy videos | No | Frames come from a small number of videos, so split by video |
| CVC-ColonDB, ETIS-Larib | 380 / 196 images with masks | No | Often used as unseen test sets |
| PolypGen | Multi-centre (6 centres) images and sequences with masks | No | Good for testing generalisation across centres |
| LDPolypVideo | Large video dataset with box annotations | No | Video-level detection |
| REAL-Colon | 60 full-procedure videos from several centres, ~2.7M frames, ~350k polyp boxes | **Yes**, with size and location | Closest to real-world second-look use (full withdrawals). Check the paper for exact polyp counts and the licence |
| Gastrointestinal Lesions in Regular Colonoscopy (Mesejo et al., 2016) | Videos of 76 lesions, white light and NBI | Lesion class (hyperplastic / serrated / adenoma); check how each was confirmed | Small; no cancers |

## Colonoscopy — restricted (application needed)

| Dataset | Content | Histology |
|---|---|---|
| SUN Colonoscopy Video Database | 100 polyps, ~49k positive + ~109k negative frames | **Yes**, with size, shape and location. Mostly adenomas; few cancers |
| ASU-Mayo Clinic polyp database | Video dataset; access by request | No |
| PICCOLO | About 3,400 images of 76 lesions, white light + NBI | **Yes**, with Paris / NICE annotations |

## Capsule endoscopy

| Dataset | Content | Histology | Notes |
|---|---|---|---|
| Kvasir-Capsule | 117 videos, ~4.7M frames, ~47k labelled (14 classes) | No | Mostly small-bowel (Olympus Endocapsule 10, EC-S10). Includes polyp class |
| Colon capsule data | Very little is public | Partner data only | **Gap.** Partner data (NHS Scotland ScotCap/CCE programme, hospitals) plus VR-Caps synthetic data. A capsule lesion classifier needs histology from follow-up colonoscopy |

## Synthetic — VR-Caps (this repository) and MADSyncro

The Unity environment in `VR-Caps-Unity/` renders colon models built from CT, with real
mucosa textures, polyps (see `img/PolypData.png`), capsule optics, lighting and depth. Its
recorder exports RGB and depth sequences. To use them for SecondLook:

1. Add polyp meshes with a dedicated material or layer, and export a segmentation (AOV)
   pass as the mask, next to the RGB frames.
2. Arrange the output as `images/` + `masks/` and pre-train with `training/train_segmentation.py`.
3. Fine-tune on real data. The VR-Caps paper showed that pre-training on synthetic data
   improves classification on Kvasir (`Tasks/Disease Classification`).
4. Use synthetic data to over-represent rare cases: flat/depressed lesions, lesions behind folds,
   poor bowel prep, capsule tumbling.
5. For lesion-classifier pre-training, give each lesion material a category (benign /
   precancerous / cancerous) and export a `labels.csv` (image, label, patient_id, mask) with one
   "patient" per rendered lesion, so its frames stay on one side of the split. Fine-tune and
   validate on real histology-labelled data. A synthetic-only classifier is never valid.
   Add `{"synthetic": true}` to a `dataset.json` in the export folder, as
   `make_synthetic_lesions.py` does, so a model trained only on it says so in every report.

`training/make_synthetic_lesions.py` writes simpler, procedural lesion images in the same
format (pale hyperplastic-like, lobulated adenoma-like with a surface pattern, ulcerated
cancer-like). They are for testing the pipeline, not a substitute for VR-Caps or real data.

## Clinical data from partners (Phase 2)

* Local study data must go through the health board's governance. In Scotland that means a Caldicott
  Guardian approval or Public Benefit and Privacy Panel (PBPP) application, a Safe Haven where
  appropriate, and a DPIA.
* Pseudonymise at source. Mask burned-in on-screen text (patient name, DOB, CHI number) before
  data leaves the trust.
* Collect histology, polyp size and segment location with each lesion. These are needed for the CADx work.
  Record the imaging mode (white light, NBI, BLI, LCI), the endoscope/processor model and the centre too,
  so the classifier can be evaluated per group.
* Collect cancers on purpose. They are rare in screening data and almost absent from public data: use
  recorded procedures of cancers confirmed at the colorectal cancer MDT and bowel screening programme
  colonoscopies with a cancer outcome. Serrated and flat or depressed lesions are also under-represented.
* Include procedures with a later PCCRC or interval cancer. They are the strongest test of a "missed lesion" tool.

## Evaluation rules

* Split by **patient** (and report results by **centre**), never by frame.
* Report per-lesion sensitivity, false positives per minute of video (colonoscopy) or per study (capsule), and time to first detection.
* Hold out at least one centre or device completely as an external test set.
* Lesion classifier: report per-class sensitivity and specificity, cancer sensitivity, neoplastic NPV,
  calibration and coverage at the abstention threshold, with patient-grouped confidence intervals.
  Take the headline figures from the deployed decision (abstention included), as the trainer prints them.
* Classifier images: keep them at their native resolution; the trainer shrinks them to the analysis working
  size itself, as the pipeline does with video frames.
  Evaluate diminutive rectosigmoid polyps separately against the ASGE PIVI thresholds
  ([DIAGNOSIS.md](DIAGNOSIS.md), section 5).
