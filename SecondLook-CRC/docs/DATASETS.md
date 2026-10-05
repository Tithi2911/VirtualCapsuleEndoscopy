# Datasets

**Always check the current licence and terms before using a dataset.** Many public
endoscopy datasets allow **research / non-commercial use only**. A model trained on them may
not be usable in a commercial product without permission from the dataset owners. Keep a
record of which datasets each released model was trained on. The ONNX sidecar's
`training_data` field and the audit log support this.

## Colonoscopy — open or by registration

| Dataset | Content | Notes |
|---|---|---|
| Kvasir-SEG | 1,000 polyp images with masks | Standard segmentation benchmark |
| HyperKvasir | ~110k images (10k labelled), 374 videos, GI findings | Upper and lower GI. Includes polyps and quality classes (bowel prep) |
| CVC-ClinicDB | 612 frames with masks, from colonoscopy videos | Frames come from a small number of videos, so split by video |
| CVC-ColonDB, ETIS-Larib | 380 / 196 images with masks | Often used as unseen test sets |
| PolypGen | Multi-centre (6 centres) images and sequences with masks | Good for testing generalisation across centres |
| LDPolypVideo | Large video dataset with box annotations | Video-level detection |
| REAL-Colon | 60 full-procedure videos, ~2.7M frames, 350+ polyps with histology | Closest to real-world second-look use (full withdrawals, histology) |

## Colonoscopy — restricted (application needed)

| Dataset | Content |
|---|---|
| SUN Colonoscopy Video Database | 100 polyps, ~49k positive + ~109k negative frames, with histology and size |
| ASU-Mayo Clinic polyp database | Video dataset; access by request |
| PICCOLO | White light + NBI, with histology |

## Capsule endoscopy

| Dataset | Content | Notes |
|---|---|---|
| Kvasir-Capsule | 117 videos, ~4.7M frames, ~47k labelled (14 classes) | Mostly small-bowel (PillCam SB3). Includes polyp class |
| Colon capsule data | Very little is public | **Gap.** Partner data (NHS Scotland ScotCap/CCE programme, hospitals) plus VR-Caps synthetic data |

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

## Clinical data from partners (Phase 2)

* Local study data must go through the health board's governance. In Scotland that means a Caldicott
  Guardian approval or Public Benefit and Privacy Panel (PBPP) application, a Safe Haven where
  appropriate, and a DPIA.
* Pseudonymise at source. Mask burned-in on-screen text (patient name, DOB, CHI number) before
  data leaves the trust.
* Collect histology, polyp size and segment location with each lesion. These are needed for the CADx work.
* Include procedures with a later PCCRC or interval cancer. They are the strongest test of a "missed lesion" tool.

## Evaluation rules

* Split by **patient** (and report results by **centre**), never by frame.
* Report per-lesion sensitivity, false positives per minute of video (colonoscopy) or per study (capsule), and time to first detection.
* Hold out at least one centre or device completely as an external test set.
