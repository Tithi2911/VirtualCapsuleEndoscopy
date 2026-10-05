# Regulatory and Governance Plan (UK → EU → global)

> This is a planning outline, not legal or regulatory advice. Regulations in this area are
> changing (UK post-market and pre-market reforms, EU AI Act). Confirm the current position with the MHRA,
> a Notified/Approved Body and a regulatory consultant before you rely on any of it.

## 1. Is it a medical device?

Yes, once used clinically. Software that analyses patient images to detect lesions and inform
diagnosis is Software as a Medical Device (SaMD). Research-only use on de-identified data
under an ethics-approved protocol is not placing a device on the market. That is how the
prototype should be used now. The intended-use notice on every output reflects this.

**Draft intended use (to refine):**
> SecondLook is software intended to assist qualified healthcare professionals in the
> retrospective review of recorded lower-GI colonoscopy and colon capsule endoscopy images and
> video, by highlighting regions that may contain colorectal lesions and indicating which were
> not documented in the procedure report. It does not replace review of the full recording
> by a clinician and is not intended for use during the procedure.

Writing it as an offline "second reader" that only *highlights* findings (not CADx, not
real-time) is the lowest-risk first claim. Each later feature (malignancy risk, real time,
autonomous capsule control) widens the intended use and needs its own conformity assessment.

### Optical diagnosis (CADx) claim

The prototype now contains a research CADx feature. It predicts whether each detected lesion
is benign, precancerous or cancerous ([DIAGNOSIS.md](DIAGNOSIS.md)). It is **not part of the
first claim** and is for research use only. It is a separate, higher-risk claim:

* A detection-only second look asks a clinician to look again. A category such as "benign" can
  directly inform a decision to leave a polyp in place, discard it without histology, or refer
  for surgery.
* EU MDR Annex VIII Rule 11 places decision-informing software in class IIa. It is IIb where a
  wrong decision could cause serious deterioration of health or a surgical intervention, and
  III where it could cause death or irreversible deterioration. A cancer called benign is
  such a risk, so the working assumption for a CADx claim is **class IIb or higher**. The
  class follows from the final intended use and must be agreed with the Notified Body
  (Approved Body for GB). MDCG 2019-11 gives guidance on qualifying and classifying software.
* It needs its own intended use, risk analysis, usability work, clinical evaluation and
  conformity assessment. Evidence should be per lesion on recorded procedures with
  histology, from external sites and devices, benchmarked against endoscopists (for
  diminutive polyps, against ASGE PIVI and current ESGE / BSG standards).

**Draft intended use for a later CADx claim (to refine):**
> SecondLook optical diagnosis is software intended to assist qualified healthcare
> professionals reviewing recorded lower-GI colonoscopy video, by giving a prediction of the
> likely histology category (non-neoplastic, neoplastic non-invasive, or suspected cancer) of
> lesions identified in the recording, with a calibrated confidence. It does not replace
> histopathological examination and is not intended to be used on its own to decide whether
> a lesion is removed, sent for histology or left in place.

## 2. Likely classification

| Market | Framework | Likely class (detection-only, assistive) |
|---|---|---|
| Great Britain | UK MDR 2002 (as amended), UKCA via an Approved Body | Check against current MHRA rules. The planned reform aligns software classification with EU Rule 11 (≥ IIa) |
| Northern Ireland / EU | EU MDR 2017/745, CE via a Notified Body | Class IIa under Rule 11 (information used for diagnostic decisions); IIb if a wrong output could cause serious deterioration |
| EU AI Act | High-risk AI (medical device that needs a Notified Body) | AI Act obligations sit on top of MDR. Check application dates for Annex I products |
| USA | FDA | Comparable colonoscopy CADe systems were cleared via De Novo / 510(k) (Class II) |

The table is for the detection-only claim. A CADx claim (above) is assumed to be class IIb or
higher in GB and the EU, and FDA routes for CADx should be checked separately.

GB has been accepting CE-marked devices for a transition period. Confirm the current
cut-off dates with MHRA, because a CE mark may be the faster route to the GB market.

## 3. What to build now so certification is achievable later

| Requirement | Standard | Where the prototype already helps |
|---|---|---|
| Quality management system | ISO 13485 | Version control, tests, versioned models |
| Software lifecycle | IEC 62304: likely Class B for the detection-only second look; the CADx software items are likely **Class C** (a cancer called benign can contribute to serious injury) unless histopathology confirmation is formally credited as a risk control external to the software, which the risk file must then justify | Modular design, unit/integration tests, documented architecture (DESIGN.md) |
| Risk management | ISO 14971 | Hazards: missed lesion (false reassurance), false flag (unneeded repeat), wrong patient/recording. Mitigations: intended-use notice, blind-segment reporting, input hashing |
| Risk management (CADx) | ISO 14971 | Hazards: cancer or adenoma called benign (false reassurance, delayed treatment); benign called cancer (unneeded referral, anxiety); automation bias, including a category shown for a false detection; use on an untested device, imaging mode or capsule data; inflated validation figures (patient leakage, train/serve resolution skew). Mitigations: off unless a model is supplied, calibrated probabilities labelled as such only when the fit succeeded, abstention (including no benign call when neoplasia is as likely), multi-frame agreement, explanation map, safety statement on every prediction, modality check, offline only, model version and file hashes in the audit log, every output stating that histopathology must confirm it |
| Usability | IEC 62366-1 | Explanation for every flag; decision capture; formative studies with trainees and consultants |
| Health software | IEC 82304-1 | Product-level requirements and documentation |
| Clinical evaluation | MDR Annex XIV / MHRA guidance | Retrospective multi-centre study (DESIGN.md Phase 2), then prospective study |
| AI-specific | MHRA "Software and AI as a Medical Device" programme; Good Machine Learning Practice principles (MHRA/FDA/Health Canada); predetermined change control plans | Model sidecar with training data and metrics; audit trail of model versions |
| Cybersecurity | IEC 81001-5-1; NHS DSPT; Cyber Essentials Plus | Offline by default; localhost-only UI; no outbound connections |

The MHRA **AI Airlock** regulatory sandbox is worth applying to: it is aimed at AI medical devices like this one.

## 4. NHS deployment (Scotland first)

* **Clinical safety:** DCB0129 (manufacturer's clinical safety case and hazard log, signed by a Clinical Safety Officer). The health board completes DCB0160. These are NHS England standards, but NHS Scotland boards usually expect the equivalent.
* **Data protection:** UK GDPR / DPA 2018, a DPIA per health board, and Caldicott Guardian sign-off. On-premises processing simplifies this a lot. Patient data never leaves the board.
* **Information security:** the health board's information security assessment; NHS Scotland procurement may use the Scottish Government's Cyber Resilience frameworks. Also prepare the NHS DTAC (Digital Technology Assessment Criteria), which most NHS buyers will ask for.
* **Research route:** NHS Research Scotland / CSO, NHS REC ethics approval and Safe Haven access for retrospective studies (DATASETS.md).
* **Evidence for buyers:** NICE Evidence Standards Framework for digital health technologies; health-economic case (cost of PCCRC compared with the cost of a second read).

## 5. Sequencing

1. **Now:** research use only, on de-identified data. Set up the QMS, risk file and software development plan. Freeze the intended use.
2. **Phase 1–2:** train and validate the detector. Retrospective clinical study under ethics approval. DCB0129 safety case.
3. **First market:** UKCA (and/or CE under EU MDR) for detection-only offline second look.
4. **Later:** CADx (benign / precancerous / cancerous optical diagnosis) as a separate, higher-class submission with its own clinical validation; capsule-specific claims likewise; FDA once there is UK/EU evidence.
