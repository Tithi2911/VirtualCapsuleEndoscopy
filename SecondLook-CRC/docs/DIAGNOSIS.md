# Optical diagnosis: benign, precancerous or cancerous

> **Research feature. Not a medical device.** The lesion classifier predicts the likely
> histology category of a lesion from endoscopic images. That is *optical diagnosis*, not a
> histological or cell-level diagnosis. Histopathology of the resected tissue remains the
> reference standard. No validated model ships with SecondLook. A model trained on the
> synthetic images in this repository has learnt drawings, not pathology.

This document covers what the feature claims, how it works, what data it needs, how to train
and evaluate a model, how the design limits harm, and what it changes for regulation.

![A finding in the report with its AI optical diagnosis](img/diagnosis_example.jpg)

*A finding from `secondlook demo --classifier` using a model trained on the synthetic drawings
(a software test, not a clinical model). It shows the category, the probability for each
category averaged over 8 frames, frame agreement, the occlusion map, the model's provenance,
the safety notice and the second reader's fields for their own optical diagnosis and the
histology result.*

## 1. Lesions, not cells

A colonoscope or capsule camera images the **surface of the bowel wall**. It shows a
lesion's size, shape, border, colour, surface (pit) pattern and vessel pattern, especially
close up or with magnification, but not individual cells. (Specialised ultra-magnifying
endocytoscopes can show nuclei; SecondLook does not use them.) Dysplasia grade, depth of invasion,
lymphovascular invasion and resection margins are judged by a pathologist under the
microscope, sometimes with special stains. That is the cell-level diagnosis.

What an AI (or an endoscopist) can do from the image is **predict the histology the
pathologist will report**. Endoscopists already do this with classifications such as NICE
(for narrow-band imaging), JNET, Kudo pit patterns and WASP for serrated lesions. SecondLook
groups the possible histologies into three categories, because three categories are what
change management:

| Category (report label) | Histology it contains | Why grouped |
|---|---|---|
| **Benign** (non-neoplastic) | Hyperplastic polyp; inflammatory polyp / pseudopolyp; normal mucosa; lymphoid aggregate | Non-neoplastic histology. Only *diminutive (5 mm or smaller) rectosigmoid* hyperplastic polyps are candidates for "leave in place" (ASGE PIVI, section 5); larger or more proximal hyperplastic polyps still matter for surveillance (see below) |
| **Precancerous** (adenoma / SSL / TSA) | Tubular, tubulovillous and villous adenoma, with low- or high-grade dysplasia; sessile serrated lesion (SSL, formerly SSA/P); traditional serrated adenoma (TSA) | Neoplastic but not invasive: removed to prevent cancer |
| **Cancerous** (suspected cancer) | Adenocarcinoma, including suspected submucosal invasion (T1) and more advanced cancer; **intramucosal carcinoma** (see below) | Changes management: tattoo, biopsy rather than piecemeal removal, staging, multidisciplinary team (MDT) referral |

**"Benign" is not the same as "outside surveillance".** UK post-polypectomy surveillance
guidance (BSG/ACPGBI/PHE, Rutter et al., *Gut* 2020) uses *serrated polyp* as an umbrella term
that includes hyperplastic polyps, and counts as *premalignant polyps* all adenomas and all
serrated polyps except diminutive (1-5 mm) rectal hyperplastic polyps; a serrated polyp of
10 mm or more is an *advanced colorectal polyp*. So most lesions this software calls
"Benign (non-neoplastic)" still count towards surveillance under that guidance, and a benign
AI call does not take a lesion out of that count. Check the current version of the guideline.

The mapping lives in `secondlook/diagnosis/taxonomy.py` (`HISTOLOGY_TO_CATEGORY`,
`category_for()`), so training labels can be written as histology text ("tubular adenoma",
"SSL", "adenocarcinoma"), with or without a dysplasia grade as pathology reports give it
("tubular adenoma, low-grade dysplasia", "SSL with dysplasia"; dysplasia never maps below
precancerous), or as category names. An unknown label stops training with a list of
the offending values rather than being guessed. Some labels are deliberately left out so that
a person decides: "neoplastic" alone could be an adenoma or a carcinoma, and a lipoma is a
benign mesenchymal neoplasm, so it fits neither "non-neoplastic" nor "precancerous" (exclude
such lesions, or pass a custom mapping and document it).

**Intramucosal carcinoma is mapped to cancerous.** Pathology services differ: many Western
services report it as high-grade dysplasia (it carries very little risk of spread in the
colon), while others call it carcinoma. The taxonomy takes the conservative choice for a
triage tool, so the AI errs towards the higher-risk category. If your pathology service
reports it as high-grade dysplasia, pass a custom mapping and record that in the model's
documentation.

**What the feature does not do:** grade dysplasia, predict depth of invasion (deep vs
superficial submucosal), stage a cancer, assess resection margins, or tell an SSL from a
hyperplastic polyp reliably. It does not report finer subtypes or Paris morphology. The
`histology_prediction` and `morphology` fields exist in the output for future models but
stay empty today.

## 2. How it works, end to end

```
 recording ─▶ quality gating ─▶ detector ─▶ tracker ─▶ finding (one per lesion, many frames)
                                                              │
                     ┌────────────────────────────────────────┘
                     ▼
   select up to 8 frames spread over the track (always including the best one)
                     │
                     ▼
   at_working_size + crop_lesion(frame, bbox, size, margin)  ◀── the same functions used in training
                     │
                     ▼
   ONNX classifier ─▶ logits ─▶ softmax(logits / T)   (T = calibrated temperature)
                     │
                     ▼
   drop non-finite outputs; average probabilities over frames ─▶ category, confidence,
                     │       frame agreement, neoplasia risk, malignancy risk (mean and peak)
                     ▼
   abstain?  top probability < abstain_below, benign but P(neoplastic) >= P(benign),
             frames disagree, several frames (for a benign call, any one frame)
             confidently higher-risk ─▶ "Indeterminate"
                     │
                     ▼
   occlusion-sensitivity map on the best frame's crop ─▶ findings/F001_diagnosis.png
                     │
                     ▼
   report.html / result.json ─▶ second reader records own diagnosis + histology ─▶ export
                     │
                     ▼
   histology-confirmed cases ─▶ labels.csv ─▶ fine-tune ─▶ external validation ─▶ new version
```

1. **Detection and tracking** (`detectors/`, `tracking.py`). The lesion classifier never
   looks for lesions itself. It only characterises findings the detector has already
   produced, so a missed lesion is never characterised, and a false detection still gets a
   category. The reviewer must first decide whether a finding is a lesion at all.
2. **Multi-frame crops** (`diagnosis/classifier.py: select_detections`,
   `diagnosis/preprocess.py: crop_lesion`). Up to `max_frames` (default 8) detections are
   taken from equal stretches of the track, so the classifier sees the lesion from different
   distances and angles rather than eight copies of one moment. Each crop is a square around
   the detection box plus a margin (`crop_margin`, default 25% of the box's larger side),
   padded rather than stretched at the image border. Training uses the **same function and
   margin**, so the model is not trained on differently framed images from the ones it
   sees in use (train/serve skew). Resolution must match too. The pipeline shrinks every
   frame to the modality's working size (512 px longest side for colonoscopy, 336 px for
   capsule) before detection, so a 120 px lesion in a 1920 x 1080 frame reaches the
   classifier as about 32 px, and surface detail finer than a few pixels is lost. The
   trainer therefore shrinks each training image to the same working size
   (`preprocess.at_working_size`, the function ingest uses) before cropping, and records it
   in the sidecar (`working_size`); the classifier applies the same step to every frame, and
   refuses to characterise a recording analysed at a smaller working size. Training on
   native-resolution crops would score the model on sharper images than it gets in use and
   overstate its performance. Classifying from native-resolution frames (re-decoding the
   selected frames) would keep more surface detail; it is on the roadmap and would need the
   same change on both sides.
3. **Classifier.** Any backbone exported to ONNX with input `image` (N, 3, S, S) float32,
   RGB, normalised with the sidecar's mean and std, and output `logits` (N, 3) in the order
   benign, precancerous, cancerous. `training/train_classifier.py` offers `tiny` (tests
   only), `resnet18`, `resnet50`, `efficientnet_b0` and `convnext_tiny`.
4. **Calibration.** Training weights the loss by inverse class frequency so rare classes
   (cancer) are learnt at all. That weighting also shifts the model's class prior: left in,
   it inflates P(cancer) on imbalanced data, and no single temperature can undo a per-class
   shift. The trainer therefore adds the logit adjustment −log(class weight) to the exported
   logits (`logit_adjustment` in `model.json`), which restores the training-set class mix.
   A network's raw softmax output is also usually over-confident, so training then fits one
   temperature T on the validation set (temperature scaling) and stores it in `model.json`.
   At inference the probabilities are `softmax(logits / T)`. The report shows them as
   numbers, not just colours. If every validation image is already classified correctly,
   no temperature can be estimated (the fit would sharpen without limit), so the trainer
   keeps T = 1 and warns: a bigger or harder validation set is needed. In that case, when
   the fit hits its search limit, or when the validation split has no example of a class
   (the temperature and the best epoch were then chosen without it), the sidecar says
   `"calibrated": false`, and the report labels the numbers "Model probability (not
   calibrated)" instead of "Probability per category, calibrated on the model's own
   validation data". A sidecar without `"calibrated": true` is always treated as
   uncalibrated. Calibration holds for the class mix of the training data; a population
   with a different mix (screening rather than symptomatic, say) shifts it. Even calibrated
   probabilities are model outputs, not risks for a given patient, and the report says so.
5. **Aggregation.** Probabilities are averaged over the frames used; frames whose model
   output is not a finite number are dropped first. The predicted category is the one with
   the highest mean probability. The finding also records the confidence (that probability),
   *frame agreement* (the fraction of frames whose own top category matches),
   *neoplasia risk* = P(precancerous) + P(cancerous), *malignancy risk* = P(cancerous) and
   the highest single-frame P(cancerous).
6. **Abstention.** The classifier gives **no category** ("Indeterminate (AI abstained)") when
   any of these holds, and shows the reason:
   * the confidence is below `abstain_below` (default 0.6; at least 0.5 is enforced by the
     trainer and the classifier);
   * the top category is benign but P(precancerous) + P(cancerous) is at least P(benign): a
     lesion the model rates as likely neoplastic as not is **never called benign**;
   * the frame agreement is below `min_frame_agreement` (default 0.5);
   * at least two frames, and at least a quarter of them, each confidently give a
     *higher-risk* category than the average (for example two of eight frames calling
     suspected cancer while the average says precancerous), because averaging would
     otherwise hide them. For a **benign** call one such frame is enough: a single frame
     confidently calling precancerous or suspected cancer withholds "benign", on a track of
     any length (colonoscopy tracks can be as short as 3 frames, capsule tracks 2);
   * the model returned NaN or infinity for any frame;
   * the model is not meant for this recording: its sidecar `modality` (default
     colonoscopy) differs from the analysis modality, or frames are analysed at a smaller
     working size than it was trained at. The model then does not run at all: every finding
     is shown as "Not applied (no AI category)" (counted with the indeterminate findings in
     `result.json`), and the CLI and report say why.

   If the classifier fails on a finding (for example an unreadable frame), that finding is
   reported as indeterminate with the error, and the rest of the analysis continues. An
   indeterminate lesion still needs a human decision; it is never treated as benign.
   `metrics.decide()` implements the per-image part of this rule, and the trainer's headline
   figures use it, so they describe the decisions the report actually shows. The report
   applies the never-benign rule again to whatever characteriser produced the category (the
   `Characteriser` interface is a plug-in point): a "benign" whose own probabilities do not
   give P(benign) > P(precancerous) + P(cancerous), or that comes without them, is shown as
   indeterminate, and `result.json` then carries no category for it either.
7. **Explanation.** For each categorised finding the classifier computes an
   occlusion-sensitivity map on the crop from the best frame. Square patches (about a
   seventh of the crop, half-overlapping) are filled with the median colour of the crop's
   outer border, which is mostly the surrounding mucosa (the crop has a 25% margin around the
   lesion), so a hidden patch looks like plain mucosa rather than a grey or black hole the
   model never saw in training. The map records how much the log-odds of the predicted
   category fall, that is how much less sure the model becomes. Log-odds are used rather than the probability because a
   probability close to 100% hardly moves when a patch is hidden, which would leave the map
   blank for the most confident predictions. Bright areas are the parts of the
   image the decision depended on. A map that lights up the background, a specular
   highlight or the image border rather than the lesion is a warning sign. The map shows
   *where*, not *why*. It cannot prove a prediction is right.
8. **Report and second reader** (`report.py`). Each finding shows the AI category (or
   "Indeterminate" / "Not characterised"), the probability table (labelled calibrated only
   when it is), frames used, frame agreement, neoplasia risk, the model name, version, file
   hash and validation status, the explanation image and the safety statement: *"AI optical
   diagnosis predicts histology from the endoscopic image; it is not a histological diagnosis
   and must not be used alone for resect-and-discard or diagnose-and-leave decisions. Confirm
   with histopathology. The AI gives a category to every detection, including ones that are
   not lesions: first decide whether the finding is a lesion at all."* An "AI lesion
   classifier" section at the top gives the model's SHA-256, training data, whether it was
   trained on synthetic images only, calibration, the warnings recorded at training and its
   intended use. A priority list puts suspected cancer first, then every finding without an
   AI category (marked for human review), then precancerous, then benign; within each group
   by the AI probability of the suspected-cancer category. The second reader records their own optical diagnosis and, when it is
   available, the histology result, and exports `review_decisions.json` together with the
   AI output. The CLI summary and the local web interface name the classifier and its
   version and repeat the short form, *"A prediction of histology, not a histological
   diagnosis: confirm with histopathology."*; the audit log records the classifier, its
   version and the SHA-256 of its two files for every analysis.
9. **Retraining.** Exported decisions with a histology result are the labelled data the
   next model needs. Turning them into `labels.csv` rows (frame images from `findings/`,
   the box, the histology text and a pseudonymous patient ID) is currently a manual step.
   A converter is on the roadmap. Retrain with `--init` from the previous `best.pt`,
   re-validate externally and release under a new version.

### Model files

A trained classifier is two files that must stay together: `model.onnx` and the sidecar
`model.json` (same name, `.json`). SecondLook refuses to load a model without its sidecar,
because the sidecar fixes the class order, preprocessing and calibration, and records
provenance.

| Key | Meaning |
|---|---|
| `name`, `version` | Shown in the report and written to the audit log |
| `task` | Must be `"lesion-classification"`; other models are refused |
| `classes` | The three categories in the model's output order (training writes `["benign", "precancerous", "cancerous"]`; another order is remapped, anything else is refused) |
| `input_size`, `mean`, `std`, `crop_margin` | Preprocessing; must match training. The ONNX input shape and output width are checked against them, with one test run, when the model is loaded |
| `modality`, `working_size` | Recordings the model is for (default colonoscopy) and the working size its training images were shrunk to; a mismatch leaves every finding indeterminate |
| `temperature`, `temperature_fitted`, `calibrated` | Temperature used, the value the fit found, and whether the fit succeeded. Only `"calibrated": true` makes the report call the probabilities calibrated |
| `class_weights`, `logit_adjustment` | Training loss weights and the offset already added to the exported logits to undo their prior shift |
| `abstain_below`, `min_frame_agreement` | Abstention thresholds (defaults 0.6 and 0.5; `abstain_below` must be at least 0.5) |
| `output` | `"logits"` or `"probabilities"` (case-insensitive); any other value is refused, and a `"probabilities"` model must return values in [0, 1] that sum to 1 |
| `backbone`, `training_data` (absolute paths), `training_datasets` (path, labels.csv SHA-256, rows, synthetic or not), `init_training_datasets` (the same for the data an `--init` checkpoint and its ancestors were trained on; the report shows them as "Fine-tuned from a model trained on"), `synthetic_training_data`, `split_sizes`, `created_utc` | Provenance |
| `metrics`, `validation_warnings` | Validation and test metrics (section 5) and the trainer's warnings, shown in the report |
| `external_validation` | Optional, added by hand after an external evaluation (for example "centre B, 2026, n=412"); shown in the report |
| `intended_use` | The research-use warning that travels with the model, shown in the report |

The sidecar carries no patient IDs or hashes of them. Salted, truncated SHA-256 hashes of
every development patient ID and image file (train, validation and test, plus what an
`--init` checkpoint was trained on) go to a separate `model.development.json`, which
`--evaluate-only` reads when it sits next to the model. Keep that file with the data
owner: a salt does not protect short IDs (hospital numbers, `P001`), which can be
recovered by hashing every possible ID.

All values are checked when the model is loaded: a NaN or out-of-range temperature or
threshold, or an unknown `output` kind, is refused rather than used. A model exported with a
fixed batch size is run in chunks of that size, the last chunk padded.

## 3. Data needed

The classifier needs lesion images whose label comes from **histopathology**, ideally with
several frames or a short clip per lesion, the lesion's location in the image (mask or box),
and a pseudonymous patient ID. Lesion size (mm), anatomical segment, imaging mode, endoscope
or processor model and centre are needed to evaluate subgroups and benchmarks (section 5).

### Public datasets

| Dataset | Per-lesion histology? | Use for this classifier |
|---|---|---|
| SUN Colonoscopy Video Database | **Yes**, plus size, shape and location | Yes. Dominated by adenomas, with few hyperplastic polyps, serrated lesions or cancers. Registration; non-commercial terms |
| PICCOLO | **Yes**, white light and NBI, with Paris / NICE annotations | Yes. About 3,400 images of 76 lesions. On request; check the terms |
| REAL-Colon | **Yes**, plus size and location, in full-procedure videos | Yes. Also the closest match to second-look use (whole withdrawals). Check the licence |
| Gastrointestinal Lesions in Regular Colonoscopy (Mesejo et al., 2016) | Lesion class (hyperplastic / serrated / adenoma); check the paper for how each was confirmed | Small (76 lesions), white light and NBI video, no cancers |
| Kvasir-SEG, HyperKvasir, CVC-ClinicDB, CVC-ColonDB, ETIS-Larib, PolypGen, LDPolypVideo | **No** (polyp present / outline only) | Detection and segmentation only. Not for this classifier |
| Kvasir-Capsule | **No** | Capsule detection only |

Two practical warnings. First, **cancers are rare in public data.** Most public polyp
collections come from screening and surveillance and contain few or no adenocarcinomas, so a
model trained only on them will have an unreliable cancer class and wide confidence
intervals. Second, some public collections share source videos or derive from each other
(for example, re-annotations of the same database), so combining them can put the same lesion
in both training and test. Check provenance before combining.

**Always check the current licence and terms** before training. Several of these datasets are
for research and non-commercial use only, and a model trained on them may not be usable in a
commercial product. The sidecar's `training_data` field records what each model used.

### Partner and targeted data (UK)

The cancer class and the real-world case mix have to come from clinical partners, under the
governance described in [DATASETS.md](DATASETS.md) (Caldicott / PBPP approval, Safe Haven,
DPIA, pseudonymisation at source, removal of burned-in patient identifiers):

* **Routine colonoscopy with linked histology**: every polyp removed or biopsied, with the
  pathology report, size, segment and imaging mode.
* **Targeted cancer collection**: recorded procedures of cancers confirmed at the colorectal
  cancer MDT, bowel screening programme colonoscopies with a cancer outcome, and endoscopy
  archives linked to pathology. Without this the cancer class stays too small to evaluate.
* **Serrated lesions and flat or depressed lesions**: under-represented everywhere and
  easy to misclassify (section 6).
* **Capsule studies with histology from follow-up colonoscopy**, for a separate capsule model.

### Dataset format (`labels.csv`)

Each dataset directory passed with `--data` contains a `labels.csv`:

| Column | Required | Meaning |
|---|---|---|
| `image` | yes | Image path relative to the directory |
| `label` | yes | Histology or category text, mapped with `category_for()` |
| `patient_id` | strongly recommended | Pseudonymous patient. The automatic split keeps all of a patient's images together |
| `mask` | optional | Lesion mask path (white = lesion); its bounding box is used |
| `x`, `y`, `w`, `h` | optional | Lesion box in pixels, instead of a mask |
| `split` | optional | `train`, `val` or `test`; overrides the automatic split |

Rows with neither mask nor box use the whole image. Patient IDs are compared ignoring case
and surrounding spaces, and the same `patient_id` in two `--data` directories is treated as
one patient, so data sets that share patients cannot leak across the split; give each
source its own ID prefix if the IDs are unrelated. A row without a `patient_id` is its own
unit, identified by the content of its image file (so identical files stay together, and a
moved or copied data set is still recognised). Other columns are
ignored by the trainer but kept useful: add `lesion_id`, `size_mm`, `segment`, `imaging_mode`, `device` and `centre`
so results can be broken down later.

```csv
image,label,patient_id,mask,x,y,w,h,split
images/p001_f0123.png,tubular adenoma,P001,masks/p001_f0123.png,,,,,
images/p001_f0131.png,tubular adenoma,P001,masks/p001_f0131.png,,,,,
images/p002_f0045.png,hyperplastic polyp,P002,,212,140,64,58,
images/p003_f0200.png,adenocarcinoma,P003,,,,,,test
```

### Synthetic lesions

`training/make_synthetic_lesions.py` draws three kinds of lesion, loosely after their
white-light appearance:

* **benign:** small, pale and smooth, with a round border and faint uniform dots;
* **precancerous:** redder and lobulated, with a brown branched surface pattern;
* **cancerous:** large and ragged, with a fibrin-covered central ulcer.

A share of lesions is drawn part-way towards a neighbouring category, as real lesions overlap
in appearance. Each synthetic patient is photographed several times. Patient IDs carry the
seed (`SYN<seed>-00001`; `--patient-prefix` sets another prefix), so sets drawn with
different seeds are never taken for the same patients. This is for testing the
software and, at most, for pre-training. The realistic synthetic source is VR-Caps and
MADSyncro renders with lesion materials labelled by category ([DATASETS.md](DATASETS.md)).
`synthetic.generate(..., lesion_kinds=["benign", "precancerous", "cancerous"])` renders a
recording with one lesion of each kind for end-to-end demonstrations; `secondlook demo
--classifier MODEL.onnx` uses it, and the test suite trains a tiny model on synthetic lesions
and runs it on such a recording (`tests/test_diagnosis.py`).

## 4. Training and use

```sh
pip install -e '.[train,model]'

# 1. Check the whole chain on synthetic drawings (minutes on a CPU). Not a clinical model.
python training/make_synthetic_lesions.py --out data/synth_lesions --n-per-class 200
python training/train_classifier.py --data data/synth_lesions --out models/cadx-synth \
    --backbone tiny --size 96 --epochs 15
#    The report will say this model was trained on synthetic images only.
#    With --classifier, the demo recording has one benign, one precancerous and one cancerous lesion.
secondlook demo --out demo_cadx --classifier models/cadx-synth/model.onnx
#    -> demo_cadx/review/report.html

# 2. Train on histology-labelled data, starting from ImageNet weights.
#    --pretrained downloads them; on an offline machine copy the torchvision file
#    (e.g. resnet18-f37072fd.pth) across and pass --weights instead.
python training/train_classifier.py --data data/piccolo --data data/sun --data data/centre_a \
    --backbone resnet18 --weights weights/resnet18-f37072fd.pth \
    --out models/lesion_cls --name crc-cadx --version 2026.10.0
#    --modality colonoscopy is the default; a capsule model needs --modality capsule and capsule data.
#    Optional: fine-tune an earlier checkpoint of the same backbone (e.g. synthetic or VR-Caps
#    pre-training, or the previous release) with --init models/<previous>/best.pt. Patients and
#    images that checkpoint was trained on (recorded as hashes in best.pt) are kept in the training
#    split, so they cannot inflate validation or test figures; a split column that puts them
#    elsewhere is refused.

# 3. External validation on a centre or device never used for training or model selection.
python training/train_classifier.py --evaluate-only models/lesion_cls/model.onnx \
    --data data/centre_b --out reports/lesion_cls-centre_b

# 4. Use it.
secondlook analyse procedure.mp4 --report procedure_report.json --classifier models/lesion_cls/model.onnx
secondlook serve --classifier models/lesion_cls/model.onnx     # local web interface
```

Training writes `model.onnx`, `model.json`, `best.pt` (checkpoint for `--init`),
`model.development.json` (hashed development patients and images), `metrics.json`,
`predictions_test.csv` and `split.json` into `--out`. **Distribute only `model.onnx` and
`model.json`.** The other files are derived from patient data: `split.json` and
`predictions_test.csv` list pseudonymous patient IDs and image paths, `best.pt` and
`model.development.json` hold salted hashes of them, and `--evaluate-only` output
(`metrics.json`, `predictions.csv`) names the evaluation data's patients too. Every reported number
comes from the exported ONNX model after calibration, that is from what will actually be
deployed. Without `--version` the version is the UTC creation time, so two training runs
never share a name and version; the report, result JSON and audit log also record the
SHA-256 of `model.onnx` and `model.json`. `--evaluate-only` writes `metrics.json` and
`predictions.csv`. It warns, and records the overlap in `metrics.json`, when the evaluation
data share a directory or an identical `labels.csv` (a copy) with the model's development
data or its `--init` checkpoint's, or, using `model.development.json`, patients (IDs
compared ignoring case) or byte-identical image files, wherever the command is run from.
Without `model.development.json` only directories and `labels.csv` files are compared; rows
without a `patient_id` are compared by image content only, which misses a re-encoded,
resized or cropped copy. It cannot certify that data are external: that depends on where
they came from. Other options: `--size` (default 224),
`--epochs`, `--batch`, `--lr`, `--crop-margin`, `--abstain-below`, `--modality`, `--seed`,
`--workers`; run with `--help` for the full list. Unknown labels, missing or unreadable
files, lesion boxes outside their image and invalid splits are all reported together, in one
error, before training starts.

## 5. Evaluation

**Split by patient, validate externally.** Frames of one lesion are near-duplicates. If they
fall on both sides of a split, scores are inflated. The automatic split (70/15/15) keeps each
`patient_id` in one part and balances the class mix; `split.json` records it. Rows without a
`patient_id` are each treated as a separate patient (identical image files excepted), which
is only safe for one image per lesion.
The validation set is used for model selection and calibration, so it is not a test set.
Claims need an **external** test set: another centre, endoscope vendor or time period,
evaluated once with `--evaluate-only`.

**Images, lesions and procedures.** The training script scores each labelled image. In use,
SecondLook classifies a *finding* from several frames. For clinical evidence, evaluate per
lesion through the full pipeline on recorded procedures with histology: per-lesion accuracy,
the effect of detector errors, and abstention rates on real video.

**Metrics produced** (`secondlook/diagnosis/metrics.py`, in `metrics.json` and the sidecar):

| Metric | Why it is there |
|---|---|
| Confusion matrix; per-class sensitivity, specificity, PPV, NPV, one-vs-rest AUROC | Shows *which* errors happen, not just how many |
| Accuracy, balanced accuracy, macro AUROC | Overall; balanced accuracy is not dominated by the commonest class |
| **Deployed decision** (`selective`, from `metrics.decide()`, abstention included): coverage, accuracy on answered lesions, a confusion matrix with an "abstained" column, neoplastic vs benign sensitivity, specificity, PPV and **NPV** of the calls actually made, the number of neoplastic lesions given no category, and where every cancer ended up (called cancer, precancerous, benign or no category) | These are the decisions the report shows. The NPV of the benign calls is what "leave in place" decisions depend on; a cancer called benign is the most harmful error. The trainer prints these as its headline figures |
| Neoplastic vs benign at P(neoplastic) >= 0.5, and cancer vs rest at the argmax, over **all** lesions including those the model abstains on; AUROC | Shows what the probabilities alone would do, without abstention |
| Expected calibration error (ECE) and reliability bins | Probabilities are shown to clinicians, so they must mean what they say |
| Coverage table for thresholds 0.5 to 0.9 | How often the model answers, and how good it is when it does. Use the table to choose `abstain_below` on validation data. Thresholds below 0.5 are not offered: they could call a lesion benign that the model rates more likely neoplastic |
| Patient-grouped bootstrap 95% confidence intervals (accuracy, balanced accuracy, macro AUROC, and the neoplastic NPV of the deployed decision). With no errors observed every resample is identical and a percentile interval would read [100%-100%]; accuracy and NPV then get an exact (Clopper-Pearson) interval with one trial per patient (6 of 6 patients correct: 54.1% to 100%), and the other metrics are marked "not estimable" | Small test sets give wide intervals; report them |

**Clinical benchmarks.** Clinicians judge optical diagnosis against published thresholds,
which are defined for specific situations:

* **ASGE PIVI** (Preservation and Incorporation of Valuable endoscopic Innovations, 2011). For
  *diminutive (5 mm or smaller) polyps* assessed with **high confidence**:
  * *Diagnose-and-leave* for suspected hyperplastic polyps in the **rectosigmoid** needs an
    NPV of at least 90% for adenomatous histology.
  * *Resect-and-discard* needs at least 90% agreement with histology-based assignment of
    post-polypectomy surveillance intervals.
* **ESGE** has published competence standards for optical diagnosis of diminutive polyps
  (position statement, 2022) and guidance on advanced imaging and computer-aided diagnosis.
  Check the current documents for their thresholds rather than relying on figures quoted
  here. UK practice also follows BSG and NICE guidance on virtual chromoendoscopy and
  surveillance; check the current versions.

These benchmarks are about two classes (adenoma vs hyperplastic), diminutive polyps,
high-confidence calls and, for diagnose-and-leave, the rectosigmoid. The three-class metrics
above do not show them on their own. To test against PIVI, evaluate the subset of 5 mm or
smaller rectosigmoid polyps (needs `size_mm` and `segment`) with the abstention threshold as
the "high confidence" rule, and compute surveillance-interval agreement with the current UK
or local surveillance guideline. Compare against endoscopists on the same cases (reader
study), and measure AI-assisted against unassisted endoscopists. Do not quote a synthetic or
internal-validation figure as meeting any benchmark.

## 6. Safety design

| Measure | What it guards against |
|---|---|
| **Optical diagnosis is opt-in** (`--classifier`). Without a model every finding is "Not characterised" | An untrained rule producing a diagnosis |
| **Calibrated probabilities, shown as numbers**, with the class-weight prior shift removed; labelled "not calibrated" whenever the temperature fit failed or the validation data lacked a class, and as model outputs rather than patient risks even when calibrated | Over-confident outputs; inflated cancer probabilities; colour-only displays; model outputs read as a patient's risk |
| **Abstention** on low confidence, frame disagreement, a benign call that is not more likely than neoplasia, a confident higher-risk minority of frames, or non-finite model output, with the reason shown | A confident-looking guess; an indeterminate or broken output being read as benign |
| **Multi-frame aggregation and frame agreement** | One unlucky frame (glare, blur, oblique view) deciding the category |
| **Model checks at load time**: sidecar values finite and in range, ONNX shapes matching the sidecar, one test run; **modality and working-size check** per recording | A misconfigured or mismatched model producing confident wrong categories; a colonoscopy model used on capsule studies |
| **Same working resolution in training and use** | A model validated on sharper crops than it receives in use |
| **Occlusion explanation** for every categorised finding | A model relying on the background, glare or image borders without anyone noticing |
| **Safety statement next to every prediction**, in the report, CLI and web interface; intended-use text inside the model sidecar | The output being taken as a histological diagnosis |
| **Conservative handling of risk order** (benign < precancerous < cancerous): ambiguous histology maps upwards (intramucosal carcinoma → cancerous); metrics headline cancer sensitivity and neoplastic NPV; uncertain predictions are withheld, never defaulted to benign | Under-calling, the error with the worst consequences |
| **Offline only**: never shown during a procedure in this version | Real-time decisions (leave in place, discard, change of technique) based on an unvalidated model |
| **Traceability**: model name, version and SHA-256 of both model files in the report, result JSON and audit log; a unique default version per training run; training data fingerprints, split, metrics, warnings and validation status in the sidecar and the report | Not knowing which model produced an output, or how far it was validated |
| **No patient leakage on retraining**: `--init` keeps the checkpoint's training patients and images in train, however the data set was moved; `--evaluate-only` detects overlap by path, `labels.csv` content (the `--init` checkpoint's data included), patient ID and image content | Inflated validation and test figures |
| **Failure isolation**: a classifier error on one finding makes that finding indeterminate and the analysis continues | One bad frame hiding the rest of the review |

### Known failure modes

* **Flat and depressed lesions** (Paris 0-IIb, 0-IIc, laterally spreading tumours) are rare in
  training data and look unlike the typical polyp. Depressed areas are a warning sign for
  invasion that the model may not have learnt.
* **Sessile serrated lesions** are pale, flat and often mucus-capped, and can look like
  hyperplastic polyps. Even pathologists disagree on SSL vs hyperplastic polyp, so the labels
  themselves are noisy.
* **Poor bowel preparation, debris, bubbles, blood, blur and glare** change the surface
  pattern the model relies on. Quality gating removes the worst frames from detection, but a
  partly obscured lesion can still be classified.
* **Imaging mode and device shift.** White light, NBI, BLI and LCI look very different, as do
  processors from different vendors and magnifying endoscopes. A model is only as valid as the
  modes and devices in its test data. Record `imaging_mode` and `device`, and report results
  per group.
* **Capsule images need their own model.** Capsule optics, resolution, lighting and the lack
  of insufflation differ from colonoscopy. Do not use a colonoscopy-trained classifier on
  capsule studies, or the reverse, without separate validation. SecondLook enforces this
  through the sidecar's `modality`: a mismatched model leaves every finding indeterminate.
* **Working resolution.** Lesions are classified at the analysis working size (512 px for
  colonoscopy). Small lesions in high-definition video lose fine surface and vessel detail
  at that size, which limits what the model can learn from them.
* **Small lesions and rare classes.** Diminutive lesions span few pixels. Cancers, TSAs and
  high-grade lesions are rare, so their metrics have wide confidence intervals.
* **Dataset and spectrum bias.** Curated still images of well-seen polyps are easier than
  real withdrawals. Case mix differs between screening, surveillance and symptomatic
  populations, and between centres and countries.
* **Detector errors propagate.** A false detection (fold, stool, bubble) still receives a
  category, and a poorly placed box gives a crop that misses part of the lesion. The report
  says so next to every AI category, but gating characterisation on detector confidence or
  adding a "not a lesion" class is still to do.
* **Synthetic training does not transfer.** The procedural lesions only test the software.
  Even realistic simulator renders need fine-tuning and validation on real data.

## 7. Regulatory impact

Characterisation (CADx) is a **higher-risk claim than detection (CADe)**. A detection-only
second look highlights regions for a clinician to check. A category such as "benign" can
directly inform decisions to leave a polyp in place, discard it without histology, or refer
a patient for surgery.

Under EU MDR Annex VIII Rule 11, software that provides information used to take diagnostic or
therapeutic decisions is class IIa. It is IIb if those decisions could cause a serious
deterioration in health or a surgical intervention, and class III if they could cause death or
an irreversible deterioration. A cancer wrongly called benign could have such consequences, so
the working assumption is that a CADx claim is **at least class IIb**. The class depends on
the final intended use and must be agreed with a Notified Body (EU) or Approved Body (GB);
check the current MHRA position for Great Britain. CADx is also high-risk AI under the EU AI
Act.

In practice this means a **separate intended use, risk analysis, clinical validation and
conformity assessment** from the detection-only second-look claim. The CADx validation should
be prospective or on a representative retrospective cohort with histology, with external
sites, and benchmarked against endoscopists. See [REGULATORY.md](REGULATORY.md). Until then
the feature is for **research use only**, on de-identified data, under an approved protocol.
